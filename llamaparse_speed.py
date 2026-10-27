"""Time how long the LlamaParse API takes to turn a PDF into markdown.

    python3 llamaparse_speed.py

Needs LLAMA_CLOUD_API_KEY in .env.
"""

import os
import time
from pathlib import Path
from time import perf_counter

import httpx
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "02_Rooms_and_Equipment_Specification.pdf"
OUTPUT_PATH = BASE_DIR / "markdown" / f"{PDF_PATH.stem}_llamaparse.md"

load_dotenv(BASE_DIR / ".env", override=True)
LLAMAPARSE_URL = "https://api.cloud.llamaindex.ai"
TIER = "cost_effective"  # fast, cost_effective, agentic or agentic_plus ("fast" gives no markdown)
headers = {"Authorization": f"Bearer {os.environ['LLAMA_CLOUD_API_KEY']}", "Accept": "application/json"}


def check(response):
    """The JSON reply, or stop with LlamaParse's error message."""
    if response.is_error:
        raise SystemExit(response.text)
    return response.json()


start = perf_counter()

# 1. Upload the PDF
files = {"file": (PDF_PATH.name, PDF_PATH.read_bytes(), "application/pdf")}
reply = check(httpx.post(f"{LLAMAPARSE_URL}/api/v1/beta/files", headers=headers,
                         data={"purpose": "parse"}, files=files, timeout=60))
file_id = reply["id"]

# 2. Start the parse job
reply = check(httpx.post(f"{LLAMAPARSE_URL}/api/v2/parse", headers=headers,
                         json={"file_id": file_id, "tier": TIER, "version": "latest"}, timeout=60))
job_id = reply["id"]

# 3. Wait until the job is done
while True:
    result = check(httpx.get(f"{LLAMAPARSE_URL}/api/v2/parse/{job_id}", headers=headers,
                             params={"expand": "markdown"}, timeout=60))
    status = result["job"]["status"]
    if status in ("COMPLETED", "FAILED", "CANCELLED"):
        break
    time.sleep(1)

if status != "COMPLETED":
    raise SystemExit(result)

pages = result["markdown"]["pages"]
markdown = "\n\n".join(f"<!-- page {number} -->\n{page['markdown']}" for number, page in enumerate(pages, start=1))
print(f"llamaparse ({TIER}): {perf_counter() - start:.3f} seconds, {len(pages)} pages, {len(markdown)} characters")

OUTPUT_PATH.write_text(markdown)
