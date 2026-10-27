"""Time how long the LiteParse server takes to turn a PDF into markdown, with OCR turned off.

    python3 liteparse_server_no_ocr_speed.py
"""

import json
from pathlib import Path
from time import perf_counter

import httpx

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "01_Club_Rules_and_Requirements.pdf"
OUTPUT_PATH = BASE_DIR / "markdown" / f"{PDF_PATH.stem}_liteparse_server_no_ocr.md"
LITEPARSE_URL = "https://docling.mariothedev.com/parse"
CONFIG = {"ocrEnabled": False, "outputFormat": "markdown"}


def parse(pdf_bytes):
    files = {"file": (PDF_PATH.name, pdf_bytes, "application/pdf")}
    response = httpx.post(LITEPARSE_URL, files=files, data={"config": json.dumps(CONFIG)}, timeout=300)
    response.raise_for_status()
    return response.json()["pages"]


pdf_bytes = PDF_PATH.read_bytes()

# The first parse may also start things up on the server
start = perf_counter()
parse(pdf_bytes)
print(f"first: {perf_counter() - start:.3f} seconds")

start = perf_counter()
pages = parse(pdf_bytes)
markdown = "\n\n".join(f"<!-- page {page['pageNum']} -->\n{page['markdown']}" for page in pages)
print(f"second: {perf_counter() - start:.3f} seconds, {len(pages)} pages, {len(markdown)} characters")

OUTPUT_PATH.write_text(markdown)
