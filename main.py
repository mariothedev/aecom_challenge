"""Step 2: upload tender PDFs, extract structured chunks, store them in PostgreSQL."""

import io
import os
import re
from pathlib import Path

import litellm
import pdfplumber
import psycopg
import pymupdf
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import PlainTextResponse
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent
# override=True: the shell may already define these names for other services.
load_dotenv(BASE_DIR / "env.development", override=True)
DATABASE_URL = os.environ["DATABASE_URL"]
EMBEDDING_TOKEN = os.environ["DEEPL_TOKEN"]

HEADER_Y = 50  # running headers sit above this point on the page
FOOTER_Y = 790  # page numbers sit below this point
MAX_HEADING_CHARS = 120  # a heading is always a short line
MAX_CHUNK_CHARS = 1800  # split longer sections so chunks stay retrievable

# DeepInfra speaks the OpenAI dialect, so LiteLLM reaches it through the openai route.
EMBEDDING_MODEL = "openai/Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_API_BASE = "https://api.deepinfra.com/v1/openai"
EMBEDDING_DIM = 1024
EMBEDDING_BATCH = 64

app = FastAPI(title="Tender contradiction detection")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


# --------------------------------------------------------------------------
# Reading the PDF
# --------------------------------------------------------------------------

def is_bold(span):
    return bool(span["flags"] & 16) or "Bold" in span["font"]


def line_text(line):
    return re.sub(r"\s+", " ", "".join(s["text"] for s in line["spans"])).strip()


def body_blocks(page):
    """Text blocks of one page, without the running header and footer."""
    for block in page.get_text("dict")["blocks"]:
        spans = [s for line in block.get("lines", []) for s in line["spans"] if s["text"].strip()]
        if not spans:
            continue
        top, bottom = block["bbox"][1], block["bbox"][3]
        if top < HEADER_Y or bottom > FOOTER_Y:
            continue
        yield block, spans


def body_size(doc):
    """The most used font size of plain text, i.e. the size of normal body copy."""
    used = {}
    for page in doc.pages(1, doc.page_count):  # skip the cover, it is all large type
        for _, spans in body_blocks(page):
            for span in spans:
                if not is_bold(span):
                    size = round(span["size"], 1)
                    used[size] = used.get(size, 0) + len(span["text"].strip())
    return max(used, key=used.get) if used else 0


def heading_size(doc, body):
    """The font size of top level headings: the largest size used by bold short lines."""
    sizes = []
    for page in doc.pages(1, doc.page_count):
        for block, spans in body_blocks(page):
            size = round(max(s["size"] for s in spans), 1)
            text = " ".join(line_text(line) for line in block["lines"]).strip()
            if is_bold(spans[0]) and size >= body and len(text) <= MAX_HEADING_CHARS:
                sizes.append(size)
    return max(sizes) if sizes else 0


def document_title(doc, filename):
    """The document's own title, printed in the largest type on the cover page."""
    lines = []
    for block in doc[0].get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = line_text(line)
            if text:
                lines.append((round(max(s["size"] for s in line["spans"]), 1), text))
    if not lines:
        return Path(filename).stem
    # The cover prints the project name first, then "Document NN", then the title itself.
    start = 0
    for index, (_, text) in enumerate(lines):
        if re.match(r"^Document\s+\d+", text):
            start = index + 1
    biggest = max(size for size, _ in lines)
    title = [text for size, text in lines[start:] if size == biggest]
    return " ".join(title) if title else Path(filename).stem


def page_pieces(page, plumber_page):
    """Everything on a page in reading order, as (text, font size, is heading candidate)."""
    tables = []
    for table in plumber_page.find_tables():
        rows = []
        for row in table.extract():
            cells = [re.sub(r"\s+", " ", cell or "").strip() for cell in row]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            tables.append((table.bbox, "\n".join(rows)))

    pieces = []
    for block, spans in body_blocks(page):
        left, top, right, bottom = block["bbox"]
        middle_x, middle_y = (left + right) / 2, (top + bottom) / 2
        # pdfplumber already captured this text as a table row, so do not repeat it.
        if any(x0 <= middle_x <= x1 and y0 <= middle_y <= y1 for (x0, y0, x1, y1), _ in tables):
            continue
        text = " ".join(line_text(line) for line in block["lines"]).strip()
        if text:
            pieces.append((top, text, round(max(s["size"] for s in spans), 1), is_bold(spans[0])))
    for (_, top, _, _), text in tables:
        pieces.append((top, text, 0, False))

    pieces.sort(key=lambda piece: piece[0])
    return [(text, size, bold) for _, text, size, bold in pieces]


# --------------------------------------------------------------------------
# Turning a document into chunks
# --------------------------------------------------------------------------

def split_text(text):
    """Cut a long section into pieces, preferring paragraph then sentence boundaries."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    pieces, current = [], ""
    for part in re.split(r"(?<=\n)|(?<=[.!?])\s+", text):
        if current and len(current) + len(part) > MAX_CHUNK_CHARS:
            pieces.append(current.strip())
            current = ""
        current += part
    if current.strip():
        pieces.append(current.strip())
    return pieces


def make_chunks(content, filename):
    """Read one PDF and return its chunks, one per section (split if very long)."""
    doc = pymupdf.open(stream=content, filetype="pdf")
    plumber = pdfplumber.open(io.BytesIO(content))

    title = document_title(doc, filename)
    body = body_size(doc)
    top_size = heading_size(doc, body)

    chunks = []
    parent, heading = "", ""
    buffer, buffer_page = [], 1

    def flush():
        if not buffer:
            return
        for text in split_text("\n".join(buffer)):
            chunks.append({
                "doc": filename,
                "page": buffer_page,
                "page_title": title,
                "section_parent": parent,
                "section_heading": heading,
                "section_content": text,
            })

    for index in range(doc.page_count):
        page_number = index + 1
        for text, size, bold in page_pieces(doc[index], plumber.pages[index]):
            if bold and size >= body and len(text) <= MAX_HEADING_CHARS:
                flush()
                buffer, buffer_page = [], page_number
                if size >= top_size:
                    parent, heading = text, ""
                else:
                    heading = text
                continue
            if not buffer:
                buffer_page = page_number
            buffer.append(text)
    flush()

    doc.close()
    plumber.close()
    return chunks


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------

def embed(texts):
    """Turn texts into vectors, a batch at a time."""
    vectors = []
    for start in range(0, len(texts), EMBEDDING_BATCH):
        answer = litellm.embedding(
            model=EMBEDDING_MODEL,
            input=texts[start : start + EMBEDDING_BATCH],
            api_base=EMBEDDING_API_BASE,
            api_key=EMBEDDING_TOKEN,
        )
        vectors += [row["embedding"] for row in answer.data]
    return vectors


def embedding_text(chunk):
    """What we actually embed: the content plus the headings that give it context.

    A clause like "At most 6 dogs may be in D-01" means little on its own; prefixing
    its document and section makes it findable by questions phrased either way.
    """
    headings = [chunk["page_title"], chunk["section_parent"], chunk["section_heading"]]
    return " > ".join(h for h in headings if h) + "\n" + chunk["section_content"]


def to_vector(values):
    """pgvector reads a vector from a literal like '[0.1,0.2]'."""
    return "[" + ",".join(map(str, values)) + "]"


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------

def ensure_schema(conn):
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS chunk_seq")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY
                DEFAULT 'chunk_' || lpad(nextval('chunk_seq')::text, 6, '0'),
            doc TEXT NOT NULL,
            page INTEGER NOT NULL,
            page_title TEXT NOT NULL DEFAULT '',
            section_parent TEXT NOT NULL DEFAULT '',
            section_heading TEXT NOT NULL DEFAULT '',
            section_content TEXT NOT NULL,
            embedding vector(%s)
        )
    """ % EMBEDDING_DIM)


def save_chunks(conn, filename, chunks, vectors):
    """Replace this document's chunks with the ones we just extracted."""
    conn.execute("DELETE FROM chunks WHERE doc = %s", (filename,))
    saved = []
    with conn.cursor() as cur:
        for chunk, vector in zip(chunks, vectors):
            cur.execute(
                """INSERT INTO chunks (doc, page, page_title, section_parent, section_heading,
                                       section_content, embedding)
                   VALUES (%(doc)s, %(page)s, %(page_title)s, %(section_parent)s,
                           %(section_heading)s, %(section_content)s, %(embedding)s::vector)
                   RETURNING chunk_id""",
                chunk | {"embedding": to_vector(vector)},
            )
            saved.append(cur.fetchone()[0])
    return saved


# --------------------------------------------------------------------------
# Web
# --------------------------------------------------------------------------

@app.get("/")
def index(request: Request):
    """Show the upload form."""
    return templates.TemplateResponse(request, "index.html")


@app.post("/uploads", response_class=PlainTextResponse)
async def uploads(files: list[UploadFile] = File(...)):
    """Chunk every uploaded PDF and store the chunks in PostgreSQL."""
    lines = []
    with psycopg.connect(DATABASE_URL) as conn:
        ensure_schema(conn)
        for file in files:
            content = await file.read()
            chunks = make_chunks(content, file.filename)
            vectors = embed([embedding_text(chunk) for chunk in chunks])
            ids = save_chunks(conn, file.filename, chunks, vectors)
            print(f"--- {file.filename}: {len(chunks)} chunks ---")
            for chunk_id, chunk in zip(ids, chunks):
                print(f"[{chunk_id}] p{chunk['page']} "
                      f"{chunk['section_parent']} > {chunk['section_heading']}")
                print(f"    {chunk['section_content'][:120]}...")
            lines.append(f"{file.filename}: {len(chunks)} chunks embedded and stored")
    return "\n".join(lines)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
