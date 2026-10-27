"""Upload tender PDFs, turn them into statements with an LLM, and store the statements with embeddings."""

import asyncio
import base64
import json
import os
from time import perf_counter
from pathlib import Path

import httpx
from anyio import fail_after
import psycopg
import pymupdf
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import numpy as np
import umap
from sklearn.cluster import HDBSCAN


BASE_DIR = Path(__file__).resolve().parent
MARKDOWN_DIR = BASE_DIR / "markdown"  # the markdown of each PDF, kept so you can check what the LLM read

# The keys in .env win over keys already set on this machine.
load_dotenv(BASE_DIR / ".env", override=True)
DATABASE_URL = os.environ["DATABASE_URL"]
DEEPINFRA_API_KEY = os.environ["DEEPINFRA_API_KEY"]
DEEPINFRA_URL = "https://api.deepinfra.com/v1/openai"
LITEPARSE_URL = "https://docling.mariothedev.com/parse"
LITEPARSE_CONFIG = {"ocrEnabled": False, "outputFormat": "markdown"}  # OCR is off: it is slow and text PDFs don't need it

CHAT_MODEL = "zai-org/GLM-5.3-Flash"
# CHAT_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct"
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

CONTRADICTION_PROMPT = """You get statements from tender documents that are about the same topic.
Find pairs that contradict each other: they cannot both be true, for example different values,
dates, ratings or quantities for the same thing, or opposite rules.
Statements that only add detail, or are about different items, are not contradictions.
If an addendum or query response changes an earlier statement, list the pair and say so in the reason.

Reply with JSON in this shape:
{"contradictions": [{"a": 12, "b": 47, "reason": "..."}]}

Use the statement ids in square brackets. Reply {"contradictions": []} if there are none."""

llm_slots = asyncio.Semaphore(10)  # at most 10 cluster checks at once


llm = ChatOpenAI(model=CHAT_MODEL, base_url=DEEPINFRA_URL, api_key=DEEPINFRA_API_KEY, temperature=0)
json_llm = llm.bind(response_format={"type": "json_object"})
# check_embedding_ctx_length=False sends plain text; otherwise LangChain sends OpenAI token ids.
embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL, base_url=DEEPINFRA_URL, api_key=DEEPINFRA_API_KEY,
                            check_embedding_ctx_length=False)

app = FastAPI(title="Tender statements")


# def convert_pdf_to_png(pdf_bytes):
#     """Each page of the PDF as a PNG image."""
#     pdf = pymupdf.open(stream=pdf_bytes, filetype="pdf")
#     pages = [page.get_pixmap(dpi=PAGE_DPI).tobytes("png") for page in pdf]
#     return pages


# async def get_markdown_from_png(png):
#     image_url = "data:image/png;base64," + base64.b64encode(png).decode()
#     message = HumanMessage(content=[
#         {"type": "text", "text": READ_PAGE_PROMPT},
#         {"type": "image_url", "image_url": {"url": image_url}},
#     ])
#     reply = await llm.ainvoke([message])
#     return reply.content


async def get_markdown_from_pdf_hetzner(document, pdf_bytes):
    files = {"file": (document, pdf_bytes, "application/pdf")}
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(LITEPARSE_URL, files=files, data={"config": json.dumps(LITEPARSE_CONFIG)})
    response.raise_for_status()
    return [page["markdown"] for page in response.json()["pages"]]


async def get_markdown_from_pdf_llamaparse(document, pdf_bytes):
    """Upload to LlamaParse, wait for parsing, and return markdown per page."""
    # url = "https://api.cloud.llamaindex.ai/api/parsing"  # retired
    url = "https://api.cloud.llamaindex.ai/api/v2/parse"
    headers = {"Authorization": f"Bearer {os.environ['LLAMA_PARSE']}"}
    # async with asyncio.timeout(300):  # requires Python 3.11+
    with fail_after(300):
        async with httpx.AsyncClient(headers=headers, timeout=60) as client:
            response = await client.post(
                f"{url}/upload",
                files={"file": (document, pdf_bytes, "application/pdf")},
                # data={"result_type": "markdown"},  # retired API
                data={"configuration": json.dumps({"tier": "agentic", "version": "latest"})},
            )
            response.raise_for_status()
            job_id = response.json()["id"]

            while True:
                # response = await client.get(f"{url}/job/{job_id}")
                response = await client.get(f"{url}/{job_id}", params={"expand": "markdown"})
                response.raise_for_status()
                result = response.json()
                # status = response.json()["status"]
                status = result["job"]["status"]
                # if status == "SUCCESS":
                if status == "COMPLETED":
                    break
                if status in {"ERROR", "FAILED", "CANCELLED"}:
                    raise RuntimeError(f"LlamaParse failed for {document}: {status}")
                await asyncio.sleep(2)

            # response = await client.get(f"{url}/job/{job_id}/result/json")
            # response.raise_for_status()
            # return [page["md"] for page in response.json()["pages"]]
            return [page["markdown"] for page in result["markdown"]["pages"]]


async def extract_statements(page):
    question = f"{page}\n\nList the statements on this page as JSON."
    reply = await json_llm.ainvoke([SystemMessage(EXTRACT_PROMPT), HumanMessage(question)])
    return json.loads(reply.content)["statements"]


async def create_table():
    async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
        await conn.execute(f"""
            CREATE EXTENSION IF NOT EXISTS vector;

            CREATE TABLE IF NOT EXISTS statements (
                id serial PRIMARY KEY,
                document text,
                page integer,
                section text,
                entity text,
                kind text,
                text text,
                embedding vector({EMBEDDING_DIM}),
                cluster_id integer
            );

            CREATE TABLE IF NOT EXISTS contradictions (
                id serial PRIMARY KEY,
                statement_a integer REFERENCES statements(id) ON DELETE CASCADE,
                statement_b integer REFERENCES statements(id) ON DELETE CASCADE,
                reason text
            );
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
    # Turn PDFs to PNGs -- skipped for now
    # pages = await asyncio.gather(*[get_markdown_from_png(png) for png in convert_pdf_to_png(pdf_bytes)])

    # LiteParse reads the PDF's text layer and gives back the markdown of each page
    start = perf_counter()
    pages = await get_markdown_from_pdf_llamaparse(document, pdf_bytes)
    print(f"LiteParse [{document}]: {perf_counter() - start:.3f} seconds")

    # Combine all pages into one markdown string per PDF
    markdown = "\n\n".join(f"<!-- page {number} -->\n{page}" for number, page in enumerate(pages, start=1))
    # Overwrites what's in file system
    MARKDOWN_DIR.mkdir(exist_ok=True)
    (MARKDOWN_DIR / f"{Path(document).stem}.md").write_text(markdown)

    # Give one markdown page to the LLM at a time
    extraction_start = perf_counter()
    statements_per_page = await asyncio.gather(*[extract_statements(page) for page in pages])
    extraction_seconds = perf_counter() - extraction_start
    statements = []
    for number, page_statements in enumerate(statements_per_page, start=1):
        for statement in page_statements:
            statements.append(statement | {"page": number})

    print(f"LLM extraction [{document}]: {len(statements)} statements in {extraction_seconds:.3f} seconds")

    # return one embedding vector per statement
    start = perf_counter()
    vectors = await embedder.aembed_documents([f"{s.get('entity')}: {s.get('text')}" for s in statements])
    print(f"Embeddings [{document}]: {perf_counter() - start:.3f} seconds")
    await save_statements(document, statements, vectors)
    print(f"{document}: {len(statements)} statements")
    return len(statements)


def cluster_vectors(vectors):
    """One cluster number per vector; -1 means it fits no cluster."""
    if len(vectors) < 3:
        return [-1] * len(vectors)
    if len(vectors) > 20:  # UMAP needs a reasonable number of points
        vectors = umap.UMAP(
            n_components=10,
            n_neighbors=min(15, len(vectors) - 1),
            metric="cosine",
            random_state=42,
        ).fit_transform(vectors)
    return HDBSCAN(min_cluster_size=2).fit_predict(vectors).tolist()


async def cluster_statements():
    async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
        cur = await conn.execute("SELECT id, embedding::text FROM statements ORDER BY id")
        rows = await cur.fetchall()
        if not rows:
            return 0

        ids = [row[0] for row in rows]
        vectors = np.array([json.loads(row[1]) for row in rows], dtype=np.float32)

        # UMAP and HDBSCAN are CPU-heavy, so keep them off the event loop
        labels = await asyncio.to_thread(cluster_vectors, vectors)

        async with conn.cursor() as cur:
            await cur.executemany(
                "UPDATE statements SET cluster_id = %s WHERE id = %s",
                list(zip(labels, ids)),
            )
    return len(set(labels) - {-1})


async def check_cluster(statements):
    """Contradicting (id_a, id_b, reason) pairs within one cluster."""
    lines = "\n".join(
        f"[{id}] ({document}, page {page}) {entity}: {text}"
        for id, document, page, entity, text in statements
    )
    async with llm_slots:
        reply = await json_llm.ainvoke([SystemMessage(CONTRADICTION_PROMPT), HumanMessage(lines)])

    ids = {s[0] for s in statements}
    pairs = []
    for c in json.loads(reply.content).get("contradictions", []):
        try:
            a, b = int(c["a"]), int(c["b"])
        except (KeyError, TypeError, ValueError):
            continue
        if a in ids and b in ids and a != b:  # ignore ids the LLM made up
            pairs.append((a, b, c.get("reason")))
    return pairs


async def find_contradictions():
    """Send each cluster to the LLM and replace the contradictions table with the result."""
    async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
        cur = await conn.execute(
            """SELECT cluster_id, id, document, page, entity, text
               FROM statements WHERE cluster_id <> -1
               ORDER BY cluster_id, id"""
        )
        rows = await cur.fetchall()

    clusters = {}
    for cluster_id, *statement in rows:
        clusters.setdefault(cluster_id, []).append(statement)

    results = await asyncio.gather(*[check_cluster(s) for s in clusters.values()])
    pairs = [pair for result in results for pair in result]

    statements_by_id = {row[1]: row[2:] for row in rows}
    print(f"Found {len(pairs)} contradiction(s)")
    # for a, b, reason in pairs[:2]:
    #     print("--- Contradiction ---")
    #     for statement_id in (a, b):
    #         document, page, entity, text = statements_by_id[statement_id]
    #         print(f"[{statement_id}] {document}, page {page} | {entity}: {text}")
    #     print(f"Reason: {reason}")

    async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
        await conn.execute("DELETE FROM contradictions")
        async with conn.cursor() as cur:
            await cur.executemany(
                "INSERT INTO contradictions (statement_a, statement_b, reason) VALUES (%s, %s, %s)",
                pairs,
            )
    return len(pairs)


@app.on_event("startup")
async def startup():
    await create_table()


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.post("/upload", response_class=HTMLResponse)
async def upload(files: list[UploadFile]):
    total_start = perf_counter()
    start = perf_counter()
    pdfs = [(file.filename, await file.read()) for file in files]
    print(f"read uploads: {perf_counter() - start:.3f} seconds")

    start = perf_counter()
    # Get Markdown from the PDFs (also fires LLM to get statements from the single markdown page)
    counts = await asyncio.gather(*[ingest(name, pdf_bytes) for name, pdf_bytes in pdfs])
    print(f"ingest: {perf_counter() - start:.3f} seconds")

    # Takes all statements from the db and cluster them, grouped by similarity using their embeddings
    start = perf_counter()
    clusters = await cluster_statements()
    print(f"clustering: {perf_counter() - start:.3f} seconds")

    # Sends each cluster to the LLM and stores the contradicting pairs
    start = perf_counter()
    contradictions = await find_contradictions()
    print(f"contradiction detection: {perf_counter() - start:.3f} seconds")

    # print(f"ingest + clustering: {perf_counter() - start:.3f} seconds")
    print(f"total: {perf_counter() - total_start:.3f} seconds")

    return f"Stored {sum(counts)} statements from {len(pdfs)} files in {clusters} clusters."


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)
