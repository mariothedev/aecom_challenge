"""Time how long docling takes to turn a PDF into markdown, with no LLM.

    pip install docling
    python3 docling_speed.py
"""

from pathlib import Path
from time import perf_counter

from docling.document_converter import DocumentConverter

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "01_Club_Rules_and_Requirements.pdf"
OUTPUT_PATH = BASE_DIR / "markdown" / "01_Club_Rules_and_Requirements_docling.md"

converter = DocumentConverter()

# The first convert also loads docling's models (and downloads them the very first time)
start = perf_counter()
converter.convert(PDF_PATH)
print(f"first: {perf_counter() - start:.3f} seconds")

# The second convert is what an upload would cost once the server is running
start = perf_counter()
result = converter.convert(PDF_PATH)
markdown = result.document.export_to_markdown()
print(f"second: {perf_counter() - start:.3f} seconds, {len(markdown)} characters")

OUTPUT_PATH.write_text(markdown)
