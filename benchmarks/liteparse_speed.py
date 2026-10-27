"""Time how long LiteParse takes to turn a PDF into markdown, with no LLM.

    pip install liteparse
    python3 liteparse_speed.py
"""

from pathlib import Path
from time import perf_counter

from liteparse import LiteParse

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "01_Club_Rules_and_Requirements.pdf"
OUTPUT_PATH = BASE_DIR / "markdown" / "01_Club_Rules_and_Requirements_liteparse.md"

parser = LiteParse(output_format="markdown")

# The first parse may also start things up (and download OCR data the very first time)
start = perf_counter()
parser.parse(str(PDF_PATH))
print(f"first: {perf_counter() - start:.3f} seconds")

# The second parse is what an upload would cost once the server is running
start = perf_counter()
result = parser.parse(str(PDF_PATH))
print(f"second: {perf_counter() - start:.3f} seconds, {len(result.pages)} pages, {len(result.text)} characters")

OUTPUT_PATH.write_text(result.text)
