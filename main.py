"""Upload tender PDFs, turn them into statements with an LLM, and store the statements with embeddings."""

import asyncio
import base64
import json
import os
from time import perf_counter
from pathlib import Path

import psycopg
import pymupdf
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

BASE_DIR = Path(__file__).resolve().parent
MARKDOWN_DIR = BASE_DIR / "markdown"  # the markdown of each PDF, kept so you can check what the LLM read

# The keys in .env win over keys already set on this machine.
load_dotenv(BASE_DIR / ".env", override=True)
DATABASE_URL = os.environ["DATABASE_URL"]
DEEPINFRA_API_KEY = os.environ["DEEPINFRA_API_KEY"]
DEEPINFRA_URL = "https://api.deepinfra.com/v1/openai"

CHAT_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIM = 1024
PAGE_DPI = 150  # sharp enough for small table text

READ_PAGE_PROMPT = """Write out this page of a tender document as markdown.
Keep every heading, list and table (as a markdown table). Copy the text word for word.
Leave out running headers, footers and page numbers.
Reply with the markdown only. Do not wrap it in a ``` code block."""

EXTRACT_PROMPT = """You read one page of a tender document, written as markdown.
List every statement on the page that could be checked against another document: requirements,
rules, quantities, ratings, dimensions, dates, durations, definitions, and changes made by addenda
or query responses.

Reply with JSON in this shape:
{"statements": [{"section": "...", "entity": "...", "kind": "...", "text": "..."}]}

- section: the nearest section heading, table name or equipment tag
- entity: what the statement is about, e.g. "Freezer FZ-01 / electrical load" or "Completion date"
- kind: one of requirement, rule, specification, date, definition, amendment, other
- text: the statement in the document's own words. For a table row, name the columns so the
  statement makes sense on its own.

Write one statement per fact. A cover page or a table of contents has no statements."""

llm = ChatOpenAI(model=CHAT_MODEL, base_url=DEEPINFRA_URL, api_key=DEEPINFRA_API_KEY, temperature=0)
json_llm = llm.bind(response_format={"type": "json_object"})
# check_embedding_ctx_length=False sends plain text; otherwise LangChain sends OpenAI token ids.
embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL, base_url=DEEPINFRA_URL, api_key=DEEPINFRA_API_KEY,
                            check_embedding_ctx_length=False)

app = FastAPI(title="Tender statements")


def convert_pdf_to_png(pdf_bytes):
    """Each page of the PDF as a PNG image."""
    pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages = [page.get_pixmap(dpi=PAGE_DPI).tobytes("png") for page in pdf]
    return pages


async def get_markdown_from_png(png):
    image_url = "data:image/png;base64," + base64.b64encode(png).decode()
    message = HumanMessage(content=[
        {"type": "text", "text": READ_PAGE_PROMPT},
        {"type": "image_url", "image_url": {"url": image_url}},
    ])
    reply = await llm.ainvoke([message])
    return reply.content


async def extract_statements(page):
    question = f"{page}\n\nList the statements on this page as JSON."
    reply = await json_llm.ainvoke([SystemMessage(EXTRACT_PROMPT), HumanMessage(question)])
    return json.loads(reply.content)["statements"]


async def create_table():
    async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS statements (
                id serial PRIMARY KEY,
                document text,
                page integer,
                section text,
                entity text,
                kind text,
                text text,
                embedding vector({EMBEDDING_DIM})
            )
        """)


async def save_statements(document, statements, vectors):
    rows = [
        (document, s.get("page"), s.get("section"), s.get("entity"), s.get("kind"), s.get("text"), str(vector))
        for s, vector in zip(statements, vectors)
    ]
    async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
        await conn.execute("DELETE FROM statements WHERE document = %s", (document,))
        async with conn.cursor() as cur:
            await cur.executemany(
                """INSERT INTO statements (document, page, section, entity, kind, text, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s::vector)""",
                rows,
            )


async def ingest(document, pdf_bytes):
    # turn PDFs to PNGs
    pages = await asyncio.gather(*[get_markdown_from_png(png) for png in convert_pdf_to_png(pdf_bytes)])
    # gives us one markdown per pdf (now actually a PNG)
    markdown = "\n\n".join(f"<!-- page {number} -->\n{page}" for number, page in enumerate(pages, start=1))
    # overwrites what's in file system
    MARKDOWN_DIR.mkdir(exist_ok=True)
    (MARKDOWN_DIR / f"{Path(document).stem}.md").write_text(markdown)

    statements_per_page = await asyncio.gather(*[extract_statements(page) for page in pages])
    statements = []
    for number, page_statements in enumerate(statements_per_page, start=1):
        for statement in page_statements:
            statements.append(statement | {"page": number})

    # return one embedding vector per statement
    vectors = await embedder.aembed_documents([f"{s.get('entity')}: {s.get('text')}" for s in statements])
    await save_statements(document, statements, vectors)
    print(f"{document}: {len(statements)} statements")
    return len(statements)


@app.on_event("startup")
async def startup():
    await create_table()


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.post("/upload", response_class=HTMLResponse)
async def upload(files: list[UploadFile]):
    pdfs = [(file.filename, await file.read()) for file in files]

    start = perf_counter()
    counts = await asyncio.gather(*[ingest(name, pdf_bytes) for name, pdf_bytes in pdfs])

    print(f"convert_pdf_to_png: {perf_counter() - start:.3f} seconds")
    # return f"Stored {sum(counts)} statements from {len(pdfs)} files."
    return f"Converted {sum(counts)} pages from {len(pdfs)} files."


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)
