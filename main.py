"""Step 2: upload tender PDFs, extract structured chunks, store them in PostgreSQL."""

import io
import os
import re
from pathlib import Path
from typing import Literal

import litellm
import pdfplumber
import psycopg
import pymupdf
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

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

CHAT_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
CHAT_API_BASE = "https://api.deepinfra.com/v1/openai"
EXTRACT_CONCURRENCY = 8  # chunks read in parallel, so a full pack does not take minutes
CANDIDATE_CHUNKS = 8  # how many nearby chunks an amendment is compared against
CANDIDATE_LIMIT = 30  # and the most statements taken from them

NEIGHBOURS = 7  # statements compared against each statement when hunting contradictions
MIN_SIMILARITY = 0.70  # below this, two statements are not about the same thing

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
            num_retries=3,  # the host returns 429 "model busy" under load
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
# Reading statements out of a chunk
# --------------------------------------------------------------------------

class Statement(BaseModel):
    """One thing the pack requires, stated about one entity."""

    entity: str = Field(description="What the statement is about, as the document names it.")
    equipment_tag: str | None = Field(
        default=None, description="Tag such as D-01, BW-D, FA-01 or G-C01, if one is given."
    )
    attribute: str = Field(
        description="The property being fixed, in snake_case, e.g. maximum_dogs, "
                    "water_bowl_capacity, gate_height, supplier, cleaning_frequency."
    )
    value: str | None = Field(
        default=None, description="The value alone, without its unit, e.g. '120', '3', 'Club Team'."
    )
    unit: str | None = Field(default=None, description="The unit alone, e.g. 'cm', 'litres', 'kW'.")
    quote: str = Field(description="The shortest run of words from the text that proves this, copied exactly.")


class StatementList(BaseModel):
    statements: list[Statement]


EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You read tender documents and record what they require.\n"
     "List every statement in the text that fixes a value, a quantity, a date, a "
     "responsibility or a rule. Split each one so that a single statement covers a "
     "single attribute of a single entity.\n"
     "Rules:\n"
     "- Quote exactly. Copy the words from the text; never paraphrase or correct them.\n"
     "- Keep value and unit apart: '120' and 'cm', not '120 cm'.\n"
     "- Use the same snake_case attribute name for the same property wherever it appears, "
     "so the same property can be compared across documents.\n"
     "- Record what the text says even when it looks wrong or disagrees with another document.\n"
     "- Skip background prose, introductions and anything that fixes nothing.\n"
     "Return an empty list if the text states no requirement."),
    ("human",
     "Document: {doc}\nPage: {page}\nSection: {section}\n\n{content}"),
])


def extract_chain():
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        base_url=CHAT_API_BASE,
        api_key=EMBEDDING_TOKEN,
        temperature=0,
        max_retries=3,
    )
    return EXTRACT_PROMPT | llm.with_structured_output(StatementList, method="json_schema")


def extract_statements(chunks):
    """Ask the model what each chunk requires. Returns (chunk_id, Statement) pairs."""
    chain = extract_chain()
    inputs = [
        {
            "doc": chunk["doc"],
            "page": chunk["page"],
            "section": " > ".join(
                h for h in (chunk["page_title"], chunk["section_parent"], chunk["section_heading"]) if h
            ),
            "content": chunk["section_content"],
        }
        for chunk in chunks
    ]
    results = chain.batch(
        inputs,
        config={"max_concurrency": EXTRACT_CONCURRENCY},
        return_exceptions=True,
    )

    found = []
    for chunk, result in zip(chunks, results):
        if isinstance(result, Exception):
            print(f"!! {chunk['chunk_id']} failed: {result}")
            continue
        for statement in result.statements:
            found.append((chunk, statement))
    return found


def quote_matches(quote, content):
    """True when the quote really is in the chunk, ignoring how whitespace fell."""
    flat = lambda text: re.sub(r"\s+", " ", text).strip().lower()
    return flat(quote) in flat(content)


# --------------------------------------------------------------------------
# Working out what an addendum has replaced
# --------------------------------------------------------------------------

class Amending(BaseModel):
    """Whether a piece of text issues a binding change to the pack."""

    is_amendment: bool = Field(
        description="True only if this text itself changes what the pack requires."
    )
    reason: str = Field(description="One short sentence saying why.")


AMENDMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Tender packs are corrected over time by addenda. Decide whether the text below "
     "is itself a binding change to the pack.\n"
     "Answer true when the text issues a change, for example by numbering itself as an "
     "addendum item and saying that a value is replaced, superseded, deleted or now reads "
     "differently.\n"
     "Answer false for everything else, including:\n"
     "- the original requirements themselves, which state rules but change nothing;\n"
     "- answers to questions, clarifications and other text that is marked as information "
     "only or that is not adopted by a numbered addendum;\n"
     "- lists of what an addendum did NOT change;\n"
     "- covers, contents tables, acknowledgements and records of meetings or visits.\n"
     "A document can hold both binding and non-binding parts, so judge this text alone, "
     "not the document it sits in."),
    ("human", "Document: {doc}\nPage: {page}\nSection: {section}\n\n{content}"),
])


class Replaced(BaseModel):
    replaced_statement_ids: list[str] = Field(
        description="Ids of the statements this amendment replaces. Empty if it replaces none."
    )
    reason: str = Field(description="One short sentence saying what was replaced and why.")


REPLACED_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "An addendum has changed the pack. You are given the amendment and a list of existing "
     "statements. Say which of those statements the amendment has replaced.\n"
     "A statement is replaced only when the amendment sets a new value for the SAME property "
     "of the SAME thing, so that the old statement must no longer be followed.\n"
     "Rules:\n"
     "- Do not replace a statement about a different property, a different room or a "
     "different piece of equipment.\n"
     "- Do not replace a statement just because the amendment mentions it, refers to its "
     "section, or says it is unchanged.\n"
     "- An earlier addendum's statement can itself be replaced by a later one.\n"
     "- When the amendment replaces nothing, return an empty list.\n"
     "- Only use ids from the list you were given."),
    ("human",
     "Amendment ({doc} p{page}, {section}):\n"
     "{entity} | {attribute} = {value} {unit}\n\"{quote}\"\n\n"
     "Existing statements:\n{candidates}"),
])


def amendment_chain():
    llm = ChatOpenAI(model=CHAT_MODEL, base_url=CHAT_API_BASE,
                     api_key=EMBEDDING_TOKEN, temperature=0, max_retries=3)
    return AMENDMENT_PROMPT | llm.with_structured_output(Amending, method="json_schema")


def replaced_chain():
    llm = ChatOpenAI(model=CHAT_MODEL, base_url=CHAT_API_BASE,
                     api_key=EMBEDDING_TOKEN, temperature=0, max_retries=3)
    return REPLACED_PROMPT | llm.with_structured_output(Replaced, method="json_schema")


def find_amendment_chunks(chunks):
    """Which chunks actually change the pack. Judged per chunk, not per document."""
    results = amendment_chain().batch(
        [
            {
                "doc": chunk["doc"],
                "page": chunk["page"],
                "section": " > ".join(
                    h for h in (chunk["page_title"], chunk["section_parent"], chunk["section_heading"]) if h
                ),
                "content": chunk["section_content"],
            }
            for chunk in chunks
        ],
        config={"max_concurrency": EXTRACT_CONCURRENCY},
        return_exceptions=True,
    )
    amending = set()
    for chunk, result in zip(chunks, results):
        if isinstance(result, Exception):
            print(f"!! {chunk['chunk_id']} amendment check failed: {result}")
            continue
        if result.is_amendment:
            amending.add(chunk["chunk_id"])
            print(f"++ amendment {chunk['chunk_id']} {chunk['doc']} p{chunk['page']}: {result.reason}")
    return amending


# --------------------------------------------------------------------------
# Finding contradictions
# --------------------------------------------------------------------------

class Contradiction(BaseModel):
    statement_a_id: str = Field(description="Id of the first statement, from the list given.")
    statement_b_id: str = Field(description="Id of the second statement, from the list given.")
    canonical_entity: str = Field(
        description="One short name for the thing both statements are about, e.g. FZ-01 or main_hall."
    )
    attribute: str = Field(description="The property they disagree about, in snake_case.")
    category: Literal["numeric", "unit", "scope", "date_arithmetic", "omission",
                      "terminology", "superseded_text", "other"]
    severity: Literal["high", "medium", "low"] = Field(
        description="How much money or risk follows from getting this wrong."
    )
    confidence: float = Field(description="How sure you are this is a real contradiction, 0 to 1.")
    explanation: str = Field(description="One or two sentences saying what disagrees and why it matters.")
    quote_a: str = Field(description="The words from statement A, copied exactly.")
    quote_b: str = Field(description="The words from statement B, copied exactly.")


class ContradictionList(BaseModel):
    contradictions: list[Contradiction]


JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You check tender documents for statements that contradict each other.\n"
     "Below are statements that are already known to be about similar things. Report only "
     "the pairs that genuinely cannot both be followed.\n"
     "Two statements contradict when they fix the SAME property of the SAME thing to values "
     "that cannot both hold.\n"
     "Before deciding, convert units and do the arithmetic. Values that are equal once "
     "converted do NOT contradict, however differently they are written.\n"
     "Do NOT report:\n"
     "- statements about different rooms, tags, items or documents;\n"
     "- a preference or a permission against a requirement; allowing something and "
     "preferring something else is not a disagreement;\n"
     "- two statements that agree, repeat each other, or simply differ in wording;\n"
     "- a spare or surplus quantity that is merely larger than a minimum.\n"
     "Use category 'superseded_text' when the two statements disagreed but an addendum has "
     "already settled it; the marked status tells you this. Use 'omission' when a list that "
     "claims to be complete is missing an item another statement requires.\n"
     "Severity is about consequence: money, safety or programme impact is high; wording and "
     "already-settled differences are low.\n"
     "Quote exactly from the statements you were given. Only use ids from the list.\n"
     "Return an empty list when nothing truly contradicts."),
    ("human", "Statements:\n{statements}"),
])


def judge_chain():
    llm = ChatOpenAI(model=CHAT_MODEL, base_url=CHAT_API_BASE,
                     api_key=EMBEDDING_TOKEN, temperature=0, max_retries=3)
    return JUDGE_PROMPT | llm.with_structured_output(ContradictionList, method="json_schema")


def statement_text(statement):
    """What we embed for a statement: enough to tell what it is about."""
    value = " ".join(part for part in (statement["value"], statement["unit"]) if part)
    return (f"{statement['entity']} | {statement['attribute']} = {value or '(none)'}\n"
            f"{statement['quote']}")


def describe_for_judge(statement):
    value = " ".join(part for part in (statement["value"], statement["unit"]) if part)
    where = f"{statement['doc']} p{statement['page']}"
    section = statement["section_heading"] or statement["section_parent"]
    if section:
        where += f", {section}"
    return (f"{statement['statement_id']} [{statement['status']}] | {statement['entity']} | "
            f"{statement['attribute']} = {value or '(none)'} | {where} | \"{statement['quote']}\"")


def build_clusters(conn, statements):
    """Group each statement with its nearest neighbours, then drop duplicate groups."""
    clusters = {}
    for statement in statements:
        rows = conn.execute(
            """SELECT statement_id, 1 - (embedding <=> %s::vector) AS similarity
               FROM statements
               WHERE statement_id <> %s AND embedding IS NOT NULL
               ORDER BY embedding <=> %s::vector LIMIT %s""",
            (statement["vector"], statement["statement_id"], statement["vector"], NEIGHBOURS),
        ).fetchall()
        group = {statement["statement_id"]}
        group.update(row[0] for row in rows if row[1] >= MIN_SIMILARITY)
        if len(group) > 1:
            clusters[frozenset(group)] = None
    return list(clusters)


def describe(statement):
    """One line per statement, as the model sees it."""
    value = " ".join(part for part in (statement["value"], statement["unit"]) if part)
    return (f"{statement['statement_id']} | {statement['entity']} | "
            f"{statement['attribute']} = {value or '(none)'} | "
            f"{statement['doc']} p{statement['page']} | \"{statement['quote']}\"")


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
    conn.execute("CREATE SEQUENCE IF NOT EXISTS statement_seq")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS statements (
            statement_id TEXT PRIMARY KEY
                DEFAULT 'statement_' || lpad(nextval('statement_seq')::text, 6, '0'),
            entity TEXT NOT NULL,
            canonical_entity TEXT,
            equipment_tag TEXT,
            attribute TEXT NOT NULL,
            value TEXT,
            unit TEXT,
            quote TEXT NOT NULL,
            source_chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
            status TEXT NOT NULL DEFAULT 'active',
            superseded_by_statement_id TEXT
        )
    """)
    conn.execute(
        "ALTER TABLE statements ADD COLUMN IF NOT EXISTS embedding vector(%s)" % EMBEDDING_DIM
    )
    conn.execute("CREATE SEQUENCE IF NOT EXISTS contradiction_seq")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS contradictions (
            contradiction_id TEXT PRIMARY KEY
                DEFAULT 'contradiction_' || lpad(nextval('contradiction_seq')::text, 6, '0'),
            statement_a_id TEXT NOT NULL REFERENCES statements(statement_id) ON DELETE CASCADE,
            statement_b_id TEXT NOT NULL REFERENCES statements(statement_id) ON DELETE CASCADE,
            canonical_entity TEXT NOT NULL DEFAULT '',
            attribute TEXT NOT NULL DEFAULT '',
            category TEXT NOT NULL,
            severity TEXT NOT NULL,
            confidence REAL NOT NULL,
            explanation TEXT NOT NULL,
            quote_a TEXT NOT NULL,
            quote_b TEXT NOT NULL,
            review_status TEXT NOT NULL DEFAULT 'unreviewed',
            review_note TEXT,
            UNIQUE (statement_a_id, statement_b_id)
        )
    """)


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


@app.post("/extract", response_class=PlainTextResponse)
def extract():
    """Read every stored chunk and record the statements it makes."""
    with psycopg.connect(DATABASE_URL) as conn:
        ensure_schema(conn)
        chunks = conn.execute(
            """SELECT chunk_id, doc, page, page_title, section_parent, section_heading, section_content
               FROM chunks ORDER BY chunk_id"""
        ).fetchall()
        columns = ["chunk_id", "doc", "page", "page_title", "section_parent",
                   "section_heading", "section_content"]
        chunks = [dict(zip(columns, row)) for row in chunks]
        if not chunks:
            return "No chunks stored yet. Upload the pack first."

        found = extract_statements(chunks)

        conn.execute("DELETE FROM statements")
        unverified = 0
        with conn.cursor() as cur:
            for chunk, statement in found:
                if not quote_matches(statement.quote, chunk["section_content"]):
                    unverified += 1
                    print(f"?? quote not found in {chunk['chunk_id']}: {statement.quote!r}")
                cur.execute(
                    """INSERT INTO statements (entity, equipment_tag, attribute, value, unit,
                                               quote, source_chunk_id)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)
                       RETURNING statement_id""",
                    (statement.entity, statement.equipment_tag, statement.attribute,
                     statement.value, statement.unit, statement.quote, chunk["chunk_id"]),
                )
                print(f"[{cur.fetchone()[0]}] {statement.entity} | {statement.attribute} "
                      f"= {statement.value} {statement.unit or ''} ({chunk['doc']} p{chunk['page']})")

    return (f"{len(found)} statements extracted from {len(chunks)} chunks"
            f"\n{unverified} quote(s) could not be found in their source chunk")


STATEMENT_SQL = """
    SELECT s.statement_id, s.entity, s.attribute, s.value, s.unit, s.quote,
           s.source_chunk_id, s.status, s.embedding, c.doc, c.page, c.page_title,
           c.section_parent, c.section_heading, c.section_content
    FROM statements s JOIN chunks c ON c.chunk_id = s.source_chunk_id
"""
STATEMENT_COLUMNS = ["statement_id", "entity", "attribute", "value", "unit", "quote",
                     "chunk_id", "status", "embedding", "doc", "page", "page_title",
                     "section_parent", "section_heading", "section_content"]


def fetch_statements(conn, where="", args=()):
    rows = conn.execute(STATEMENT_SQL + where, args).fetchall()
    return [dict(zip(STATEMENT_COLUMNS, row)) for row in rows]


@app.post("/supersede", response_class=PlainTextResponse)
def supersede():
    """Find the addenda, and mark the statements they have replaced."""
    with psycopg.connect(DATABASE_URL) as conn:
        ensure_schema(conn)
        statements = fetch_statements(conn, " ORDER BY c.doc, c.page, s.statement_id")
        if not statements:
            return "No statements yet. Run Extract first."

        # Only the chunks that carry statements are worth judging.
        seen, chunks = set(), []
        for statement in statements:
            if statement["chunk_id"] not in seen:
                seen.add(statement["chunk_id"])
                chunks.append(statement)
        amending_chunks = find_amendment_chunks(chunks)
        amendments = [s for s in statements if s["chunk_id"] in amending_chunks]

        # Re-running must start from a clean slate.
        conn.execute("UPDATE statements SET status = 'active', superseded_by_statement_id = NULL")

        # An amendment can only replace what already existed when it was issued, so each
        # one is given a position and may not touch itself or any later amendment.
        position = {}
        for index, chunk_id in enumerate(dict.fromkeys(a["chunk_id"] for a in amendments)):
            position[chunk_id] = index

        replaced_count = 0
        chain = replaced_chain()
        # Oldest amendment first, so a later addendum overwrites an earlier one's decision.
        for amendment in amendments:
            mine = position[amendment["chunk_id"]]
            later = [chunk_id for chunk_id, index in position.items() if index >= mine]
            vector = to_vector(embed([describe(amendment)])[0])
            candidates = fetch_statements(
                conn,
                """ WHERE s.source_chunk_id <> ALL(%s)
                    ORDER BY (SELECT embedding FROM chunks WHERE chunk_id = s.source_chunk_id)
                             <=> %s::vector
                    LIMIT %s""",
                (later, vector, CANDIDATE_LIMIT),
            )
            if not candidates:
                continue

            by_id = {c["statement_id"]: c for c in candidates}
            answer = chain.invoke({
                "doc": amendment["doc"],
                "page": amendment["page"],
                "section": amendment["section_heading"] or amendment["section_parent"],
                "entity": amendment["entity"],
                "attribute": amendment["attribute"],
                "value": amendment["value"] or "",
                "unit": amendment["unit"] or "",
                "quote": amendment["quote"],
                "candidates": "\n".join(describe(c) for c in candidates),
            })

            for statement_id in answer.replaced_statement_ids:
                if statement_id not in by_id:
                    continue  # the model invented an id
                conn.execute(
                    """UPDATE statements SET status = 'superseded',
                                             superseded_by_statement_id = %s
                       WHERE statement_id = %s""",
                    (amendment["statement_id"], statement_id),
                )
                replaced_count += 1
                old = by_id[statement_id]
                print(f"-- {statement_id} ({old['entity']} {old['attribute']}="
                      f"{old['value']}) superseded by {amendment['statement_id']}")

        active = conn.execute(
            "SELECT count(*) FROM statements WHERE status = 'active'"
        ).fetchone()[0]

    return (f"{len(amendments)} amending statement(s) found in {len(amending_chunks)} chunk(s)"
            f"\n{replaced_count} statement(s) marked superseded"
            f"\n{active} statement(s) still active")


SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


@app.post("/scan")
def scan():
    """Compare statements with one another and record the contradictions."""
    with psycopg.connect(DATABASE_URL) as conn:
        ensure_schema(conn)
        statements = fetch_statements(conn, " ORDER BY s.statement_id")
        if not statements:
            return JSONResponse({"contradictions": [], "message": "No statements yet. Run Extract first."})

        # Statements need their own vectors so we can ask which ones talk about the same thing.
        missing = [s for s in statements if s["embedding"] is None]
        if missing:
            vectors = embed([statement_text(s) for s in missing])
            with conn.cursor() as cur:
                for statement, vector in zip(missing, vectors):
                    cur.execute("UPDATE statements SET embedding = %s::vector WHERE statement_id = %s",
                                (to_vector(vector), statement["statement_id"]))
            statements = fetch_statements(conn, " ORDER BY s.statement_id")

        by_id = {s["statement_id"]: s for s in statements}
        for statement in statements:
            statement["vector"] = statement["embedding"]
        clusters = build_clusters(conn, statements)
        print(f"{len(clusters)} cluster(s) to judge")

        results = judge_chain().batch(
            [{"statements": "\n".join(describe_for_judge(by_id[i]) for i in sorted(cluster))}
             for cluster in clusters],
            config={"max_concurrency": EXTRACT_CONCURRENCY},
            return_exceptions=True,
        )

        conn.execute("DELETE FROM contradictions")
        seen, skipped = set(), 0
        with conn.cursor() as cur:
            for cluster, result in zip(clusters, results):
                if isinstance(result, Exception):
                    print(f"!! cluster judging failed: {result}")
                    continue
                for found in result.contradictions:
                    a, b = found.statement_a_id, found.statement_b_id
                    if a not in by_id or b not in by_id or a == b:
                        skipped += 1
                        continue
                    pair = tuple(sorted((a, b)))
                    if pair in seen:
                        continue  # the same pair can surface from two overlapping clusters
                    # The quotes must really be in the statements they are attributed to.
                    if not (quote_matches(found.quote_a, by_id[a]["quote"])
                            and quote_matches(found.quote_b, by_id[b]["quote"])):
                        skipped += 1
                        print(f"?? unverified quotes for {a}/{b}")
                        continue
                    seen.add(pair)
                    cur.execute(
                        """INSERT INTO contradictions (statement_a_id, statement_b_id,
                               canonical_entity, attribute, category, severity, confidence,
                               explanation, quote_a, quote_b)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (pair[0], pair[1], found.canonical_entity, found.attribute,
                         found.category, found.severity, min(max(found.confidence, 0.0), 1.0),
                         found.explanation,
                         found.quote_a if pair[0] == a else found.quote_b,
                         found.quote_b if pair[0] == a else found.quote_a),
                    )

        rows = conn.execute("""
            SELECT contradiction_id, statement_a_id, statement_b_id, canonical_entity, attribute,
                   category, severity, confidence, explanation, quote_a, quote_b,
                   review_status, review_note
            FROM contradictions
        """).fetchall()

    columns = ["contradiction_id", "statement_a_id", "statement_b_id", "canonical_entity",
               "attribute", "category", "severity", "confidence", "explanation",
               "quote_a", "quote_b", "review_status", "review_note"]
    findings = [dict(zip(columns, row)) for row in rows]
    # Severity first, then confidence: a certain cosmetic issue stays below an uncertain costly one.
    findings.sort(key=lambda f: (SEVERITY_ORDER.get(f["severity"], 3), -f["confidence"]))
    print(f"{len(findings)} contradiction(s) stored, {skipped} rejected")
    return JSONResponse({"contradictions": findings})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)
