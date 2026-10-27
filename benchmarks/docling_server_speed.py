"""Time how long the docling server takes to turn a PDF into markdown.

    python3 docling_server_speed.py
"""

from pathlib import Path
from time import perf_counter

import httpx

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "01_Club_Rules_and_Requirements.pdf"
OUTPUT_PATH = BASE_DIR / "markdown" / "01_Club_Rules_and_Requirements_docling_server.md"
DOCLING_URL = "https://docling.mariothedev.com/v1/convert/file"


def convert(pdf_bytes):
    files = {"files": (PDF_PATH.name, pdf_bytes, "application/pdf")}
    response = httpx.post(DOCLING_URL, files=files, data={"to_formats": "md"}, timeout=300)
    response.raise_for_status()
    return response.json()


pdf_bytes = PDF_PATH.read_bytes()

# The first convert may also load docling's models on the server
start = perf_counter()
result = convert(pdf_bytes)
print(f"first: {perf_counter() - start:.3f} seconds (server: {result['processing_time']:.3f})")

start = perf_counter()
result = convert(pdf_bytes)
markdown = result["document"]["md_content"]
print(f"second: {perf_counter() - start:.3f} seconds (server: {result['processing_time']:.3f}), "
      f"{len(markdown)} characters")

OUTPUT_PATH.write_text(markdown)
