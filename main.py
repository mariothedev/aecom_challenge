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
# No longer used: nearest-neighbour pairs replaced UMAP + HDBSCAN (importing umap alone took ~8 seconds)
# import umap
# from sklearn.cluster import HDBSCAN


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

# Old prompt: the LLM got one cluster of statements and had to find the contradicting pairs itself.
# CONTRADICTION_PROMPT = """You get statements from tender documents that are about the same topic.
# Find pairs that contradict each other: they cannot both be true, for example different values,
# dates, ratings or quantities for the same thing, or opposite rules.
# Statements that only add detail, or are about different items, are not contradictions.
# If an addendum or query response changes an earlier statement, list the pair and say so in the reason.
#
# Reply with JSON in this shape:
# {"contradictions": [{"a": 12, "b": 47, "reason": "..."}]}
#
# Use the statement ids in square brackets. Reply {"contradictions": []} if there are none."""

# New prompt: the LLM gets numbered pairs and only says which pairs contradict. Same rules as before.
PAIRS_PROMPT = """You get numbered pairs of statements from tender documents. The two statements in a pair are about similar topics.
For each pair, decide if the two statements contradict each other: they cannot both be true, for example different values,
dates, ratings or quantities for the same thing, or opposite rules.
Statements that only add detail, or are about different items, are not contradictions.
If an addendum or query response changes an earlier statement, list the pair and say so in the reason.

Reply with JSON in this shape:
{"contradictions": [{"a": 12, "b": 47, "reason": "..."}]}

Use the statement ids in square brackets. Reply {"contradictions": []} if there are none."""
# First try asked for pair numbers ({"pair": 3, ...}), but the LLM sometimes mixed up pairs
# ("Pair 22 says ... but Pair 27 says ...") and the finding was stored on the wrong statements.
# Statement ids say exactly which two statements contradict.
# {"contradictions": [{"pair": 3, "reason": "..."}]}
#
# Use the pair numbers. Reply {"contradictions": []} if there are none."""

NEIGHBOURS = 5  # each statement is checked against its 5 most similar statements
MIN_SIMILARITY = 0.6  # less similar than this (cosine) and two statements are about different things
PAIRS_PER_CALL = 30  # how many pairs the LLM checks in one call

# llm_slots = asyncio.Semaphore(10)  # at most 10 cluster checks at once
llm_slots = asyncio.Semaphore(10)  # at most 10 LLM calls checking pairs at once


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


async def extract_and_embed(page, number):
    """The statements on one page and their embedding vectors.

    The page embeds its own statements as soon as its LLM extraction is done.
    So while the slowest page is still with the LLM, the faster pages are already embedded."""
    statements = await extract_statements(page)
    statements = [statement | {"page": number} for statement in statements]
    if not statements:  # a cover page or table of contents has nothing to embed
        return [], []
    vectors = await embedder.aembed_documents([f"{s.get('entity')}: {s.get('text')}" for s in statements])
    return statements, vectors


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

    # Before: extract every page, wait for ALL pages, then embed everything in one call.
    # The embedding call could only start after the slowest page was done, so its time was added on top.
    #
    # # Give one markdown page to the LLM at a time
    # extraction_start = perf_counter()
    # statements_per_page = await asyncio.gather(*[extract_statements(page) for page in pages])
    # extraction_seconds = perf_counter() - extraction_start
    # statements = []
    # for number, page_statements in enumerate(statements_per_page, start=1):
    #     for statement in page_statements:
    #         statements.append(statement | {"page": number})
    #
    # print(f"LLM extraction [{document}]: {len(statements)} statements in {extraction_seconds:.3f} seconds")
    #
    # # return one embedding vector per statement
    # start = perf_counter()
    # vectors = await embedder.aembed_documents([f"{s.get('entity')}: {s.get('text')}" for s in statements])
    # print(f"Embeddings [{document}]: {perf_counter() - start:.3f} seconds")

    # Now: every page is extracted AND embedded in its own task, all pages at once.
    # Embedding time hides behind the slowest page's LLM call instead of adding to the total.
    # Trade-off: one embedding request per page instead of one per PDF (fine for DeepInfra, but more requests).
    start = perf_counter()
    results = await asyncio.gather(*[extract_and_embed(page, number) for number, page in enumerate(pages, start=1)])
    statements = [statement for page_statements, _ in results for statement in page_statements]
    vectors = [vector for _, page_vectors in results for vector in page_vectors]
    print(f"LLM extraction + embeddings [{document}]: {len(statements)} statements in {perf_counter() - start:.3f} seconds")

    await save_statements(document, statements, vectors)
    print(f"{document}: {len(statements)} statements")
    return len(statements)


# Old: UMAP + HDBSCAN put the statements into clusters, then one LLM call checked each cluster.
# Statements HDBSCAN marked as noise (-1) were never checked, and UMAP took ~10 seconds to compile
# on its first run. Replaced by the nearest-neighbour pairs below.
#
# def cluster_vectors(vectors):
#     """One cluster number per vector; -1 means it fits no cluster."""
#     if len(vectors) < 3:
#         return [-1] * len(vectors)
#     if len(vectors) > 20:  # UMAP needs a reasonable number of points
#         vectors = umap.UMAP(
#             n_components=10,
#             n_neighbors=min(15, len(vectors) - 1),
#             metric="cosine",
#             random_state=42,
#         ).fit_transform(vectors)
#     return HDBSCAN(min_cluster_size=2).fit_predict(vectors).tolist()


# async def cluster_statements():
#     async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
#         cur = await conn.execute("SELECT id, embedding::text FROM statements ORDER BY id")
#         rows = await cur.fetchall()
#         if not rows:
#             return 0

#         ids = [row[0] for row in rows]
#         vectors = np.array([json.loads(row[1]) for row in rows], dtype=np.float32)

#         # UMAP and HDBSCAN are CPU-heavy, so keep them off the event loop
#         labels = await asyncio.to_thread(cluster_vectors, vectors)

#         async with conn.cursor() as cur:
#             await cur.executemany(
#                 "UPDATE statements SET cluster_id = %s WHERE id = %s",
#                 list(zip(labels, ids)),
#             )
#     return len(set(labels) - {-1})


# async def check_cluster(statements):
#     """Contradicting (id_a, id_b, reason) pairs within one cluster."""
#     lines = "\n".join(
#         f"[{id}] ({document}, page {page}) {entity}: {text}"
#         for id, document, page, entity, text in statements
#     )
#     async with llm_slots:
#         reply = await json_llm.ainvoke([SystemMessage(CONTRADICTION_PROMPT), HumanMessage(lines)])

#     ids = {s[0] for s in statements}
#     pairs = []
#     for c in json.loads(reply.content).get("contradictions", []):
#         try:
#             a, b = int(c["a"]), int(c["b"])
#         except (KeyError, TypeError, ValueError):
#             continue
#         if a in ids and b in ids and a != b:  # ignore ids the LLM made up
#             pairs.append((a, b, c.get("reason")))
#     return pairs


# async def find_contradictions():
#     """Send each cluster to the LLM and replace the contradictions table with the result."""
#     async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
#         cur = await conn.execute(
#             """SELECT cluster_id, id, document, page, entity, text
#                FROM statements WHERE cluster_id <> -1
#                ORDER BY cluster_id, id"""
#         )
#         rows = await cur.fetchall()

#     clusters = {}
#     for cluster_id, *statement in rows:
#         clusters.setdefault(cluster_id, []).append(statement)

#     results = await asyncio.gather(*[check_cluster(s) for s in clusters.values()])
#     pairs = [pair for result in results for pair in result]

#     statements_by_id = {row[1]: row[2:] for row in rows}
#     print(f"Found {len(pairs)} contradiction(s)")
#     # for a, b, reason in pairs[:2]:
#     #     print("--- Contradiction ---")
#     #     for statement_id in (a, b):
#     #         document, page, entity, text = statements_by_id[statement_id]
#     #         print(f"[{statement_id}] {document}, page {page} | {entity}: {text}")
#     #     print(f"Reason: {reason}")

#     async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
#         await conn.execute("DELETE FROM contradictions")
#         async with conn.cursor() as cur:
#             await cur.executemany(
#                 "INSERT INTO contradictions (statement_a, statement_b, reason) VALUES (%s, %s, %s)",
#                 pairs,
#             )
#     return len(pairs)


# New: nearest-neighbour pairs.
# A contradiction is always between two statements, and those two statements are about the same thing,
# so their embeddings are close. For every statement we take its few closest statements and ask the LLM
# about each pair. Every statement gets checked (nothing is left out as noise), and it takes milliseconds.

def find_candidate_pairs(ids, vectors):
    """Pairs of statement ids worth checking: every statement with its most similar statements."""
    if len(ids) < 2:
        return []
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # length 1, so a dot product is the cosine similarity
    similarity = vectors @ vectors.T  # similarity[i, j]: how close statements i and j are (1 = same meaning)
    np.fill_diagonal(similarity, -1)  # a statement is not its own neighbour

    pairs = set()
    for i in range(len(ids)):
        closest = np.argsort(similarity[i])[::-1][:NEIGHBOURS]
        for j in closest:
            if similarity[i, j] >= MIN_SIMILARITY:
                pairs.add((min(ids[i], ids[j]), max(ids[i], ids[j])))  # (a, b) and (b, a) are the same pair
    return sorted(pairs)  # sorted by id, so pairs that share a statement end up in the same LLM call


async def check_pairs(pairs, statements_by_id):
    """The pairs in this batch that contradict, as (id_a, id_b, reason)."""
    lines = []
    for number, (a, b) in enumerate(pairs, start=1):
        lines.append(f"Pair {number}")
        for statement_id in (a, b):
            document, page, entity, text = statements_by_id[statement_id]
            # lines.append(f"- ({document}, page {page}) {entity}: {text}")
            lines.append(f"- [{statement_id}] ({document}, page {page}) {entity}: {text}")
    async with llm_slots:
        reply = await json_llm.ainvoke([SystemMessage(PAIRS_PROMPT), HumanMessage("\n".join(lines))])

    # found = []
    # for c in json.loads(reply.content).get("contradictions", []):
    #     try:
    #         number = int(c["pair"])
    #     except (KeyError, TypeError, ValueError):
    #         continue
    #     if 1 <= number <= len(pairs):  # ignore pair numbers the LLM made up
    #         a, b = pairs[number - 1]
    #         found.append((a, b, c.get("reason")))
    # return found

    ids = {statement_id for pair in pairs for statement_id in pair}
    found = []
    for c in json.loads(reply.content).get("contradictions", []):
        try:
            a, b = int(c["a"]), int(c["b"])
        except (KeyError, TypeError, ValueError):
            continue
        if a in ids and b in ids and a != b:  # ignore ids the LLM made up
            found.append((a, b, c.get("reason")))
    return found


async def find_contradictions():
    """Check every statement against its closest statements and replace the contradictions table with the result."""
    # One connection for the whole step: connecting to the database takes about a second
    async with await psycopg.AsyncConnection.connect(DATABASE_URL) as conn:
        cur = await conn.execute("SELECT id, document, page, entity, text, embedding::text FROM statements ORDER BY id")
        rows = await cur.fetchall()
        ids = [row[0] for row in rows]
        vectors = np.array([json.loads(row[5]) for row in rows], dtype=np.float32)
        statements_by_id = {row[0]: row[1:5] for row in rows}

        start = perf_counter()
        pairs = find_candidate_pairs(ids, vectors)
        print(f"candidate pairs: {len(pairs)} in {perf_counter() - start:.3f} seconds")

        # Many pairs per LLM call, all calls at once
        batches = [pairs[i:i + PAIRS_PER_CALL] for i in range(0, len(pairs), PAIRS_PER_CALL)]
        results = await asyncio.gather(*[check_pairs(batch, statements_by_id) for batch in batches])
        # contradictions = [pair for result in results for pair in result]
        # A statement can be in more than one LLM call, so the same pair can come back twice: keep it once
        unique = {}
        for a, b, reason in [pair for result in results for pair in result]:
            unique.setdefault((min(a, b), max(a, b)), reason)
        contradictions = [(a, b, reason) for (a, b), reason in unique.items()]
        print(f"Found {len(contradictions)} contradiction(s) in {len(batches)} LLM calls")

        await conn.execute("DELETE FROM contradictions")
        async with conn.cursor() as cur:
            await cur.executemany(
                "INSERT INTO contradictions (statement_a, statement_b, reason) VALUES (%s, %s, %s)",
                contradictions,
            )
    return len(contradictions)


@app.on_event("startup")
async def startup():
    await create_table()
    # Warm up: the first embedding call is slow (setting up the HTTPS connection, DeepInfra waking the model).
    # Doing it here keeps that cost out of the first upload.
    # Note: if the server sits idle for a long time, DeepInfra can put the model to sleep again.
    await embedder.aembed_query("warm up")


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

    # No clustering step any more: find_contradictions() picks the pairs to check by itself
    # # Takes all statements from the db and cluster them, grouped by similarity using their embeddings
    # start = perf_counter()
    # clusters = await cluster_statements()
    # print(f"clustering: {perf_counter() - start:.3f} seconds")

    # Pairs every statement with its closest statements, asks the LLM about each pair, stores the contradictions
    start = perf_counter()
    contradictions = await find_contradictions()
    print(f"contradiction detection: {perf_counter() - start:.3f} seconds")

    # print(f"ingest + clustering: {perf_counter() - start:.3f} seconds")
    print(f"total: {perf_counter() - total_start:.3f} seconds")

    # return f"Stored {sum(counts)} statements from {len(pdfs)} files in {clusters} clusters."
    return f"Stored {sum(counts)} statements from {len(pdfs)} files and found {contradictions} contradiction(s)."


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)
