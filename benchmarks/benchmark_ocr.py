"""Time PDF-to-markdown services on the same PDF.

    python benchmarks/benchmark_ocr.py

liteparse: our LiteParse server, with OCR off like main.py
docling: our docling server, default settings (OCR on)
watsonx: watsonx.ai text extraction (IBM), standard mode
llamaparse: LlamaParse cloud, cost_effective tier, cache off so every run really parses
mistral-ocr-4-0: Mistral OCR on Azure

Each service's markdown is saved to markdown/benchmark_ocr/ so you can compare them.
Needs DOCLING_API_KEY, WATSONX_API_KEY, LLAMA_CLOUD_API_KEY and AZURE_API_KEY in .env.
"""

import base64
import json
import os
import re
import time
from pathlib import Path
from statistics import median
from time import perf_counter

import httpx
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

PDF_PATH = BASE_DIR / "data" / "01_Club_Rules_and_Requirements.pdf"
OUTPUT_DIR = BASE_DIR / "markdown" / "benchmark_ocr"
LITEPARSE_URL = "https://liteparse.mariothedev.com/parse"
DOCLING_URL = "https://docling.mariothedev.com/v1/convert/file"
WATSONX_URL = "https://eu-de.ml.cloud.ibm.com"
WATSONX_PROJECT_ID = "9e48aa77-4089-41a5-aad7-d6aae062ce85"
# The watsonx project's own storage bucket: watsonx reads the PDF from it and writes the markdown back to it
WATSONX_BUCKET_URL = "https://s3.eu-de.cloud-object-storage.appdomain.cloud/mariossandbox-donotdelete-pr-b18vpwbnxsqqz5"
LLAMAPARSE_URL = "https://api.cloud.llamaindex.ai/api"
AZURE_MISTRAL_URL = "https://heyaudioapp-5952-resource.services.ai.azure.com/providers/mistral/azure/ocr"
PAGE_BREAK = "<!-- page break -->"
RUNS = 3


def check(response):
    """The JSON reply, or stop with the server's error message."""
    if response.is_error:
        raise SystemExit(f"{response.url}: {response.text}")
    return response.json()


def parse_with_liteparse(pdf_bytes):
    files = {"file": (PDF_PATH.name, pdf_bytes, "application/pdf")}
    config = {"ocrEnabled": False, "outputFormat": "markdown"}
    reply = check(httpx.post(LITEPARSE_URL, files=files, data={"config": json.dumps(config)}, timeout=300))
    return [page["markdown"] for page in reply["pages"]]


def parse_with_docling(pdf_bytes):
    # docling gives one markdown for the whole PDF, so ask it to mark where each page ends
    files = {"files": (PDF_PATH.name, pdf_bytes, "application/pdf")}
    data = {"to_formats": "md", "md_page_break_placeholder": PAGE_BREAK}
    headers = {"X-Api-Key": os.environ["DOCLING_API_KEY"]}
    reply = check(httpx.post(DOCLING_URL, headers=headers, files=files, data=data, timeout=300))
    return reply["document"]["md_content"].split(PAGE_BREAK)


def get_watsonx_token():
    """watsonx wants a short-lived IAM token, made from the API key."""
    reply = check(httpx.post("https://iam.cloud.ibm.com/identity/token", timeout=60, data={
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": os.environ["WATSONX_API_KEY"]}))
    return reply["access_token"]


watsonx_headers = {"Authorization": f"Bearer {get_watsonx_token()}"}  # made once, so it is not timed


def parse_with_watsonx(pdf_bytes):
    input_file = f"speed_test/{PDF_PATH.name}"
    result_file = f"speed_test/{PDF_PATH.stem}_watsonx.md"
    version = {"version": "2024-05-01"}

    # 1. Put the PDF in the project's storage
    response = httpx.put(f"{WATSONX_BUCKET_URL}/{input_file}", content=pdf_bytes, timeout=60,
                         headers=watsonx_headers | {"Content-Type": "application/pdf"})
    if response.is_error:
        raise SystemExit(f"{response.url}: {response.text}")

    # 2. Start the extraction job
    job = {
        "project_id": WATSONX_PROJECT_ID,
        "document_reference": {"type": "container", "location": {"path": input_file}},
        "results_reference": {"type": "container", "location": {"path": result_file}},
        "parameters": {"requested_outputs": ["md"], "mode": "standard"},
    }
    reply = check(httpx.post(f"{WATSONX_URL}/ml/v1/text/extractions", params=version, headers=watsonx_headers,
                             json=job, timeout=60))

    # 3. Wait until the job is done
    while True:
        reply = check(httpx.get(f"{WATSONX_URL}/ml/v1/text/extractions/{reply['metadata']['id']}",
                                params=version | {"project_id": WATSONX_PROJECT_ID}, headers=watsonx_headers,
                                timeout=60))
        if reply["entity"]["results"]["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)

    if reply["entity"]["results"]["status"] == "failed":
        raise SystemExit(reply["entity"]["results"])

    # 4. Download the markdown. watsonx puts "<!-- Page N -->" after each page, so the last piece is empty
    response = httpx.get(f"{WATSONX_BUCKET_URL}/{result_file}", headers=watsonx_headers, timeout=60)
    if response.is_error:
        raise SystemExit(f"{response.url}: {response.text}")
    return re.split(r"<!-- Page \d+ -->", response.text)[:-1]


def parse_with_llamaparse(pdf_bytes):
    headers = {"Authorization": f"Bearer {os.environ['LLAMA_CLOUD_API_KEY']}"}

    # 1. Upload the PDF
    files = {"file": (PDF_PATH.name, pdf_bytes, "application/pdf")}
    reply = check(httpx.post(f"{LLAMAPARSE_URL}/v1/beta/files", headers=headers, data={"purpose": "parse"},
                             files=files, timeout=60))

    # 2. Start the parse job
    job = {"file_id": reply["id"], "tier": "cost_effective", "version": "latest", "disable_cache": True}
    reply = check(httpx.post(f"{LLAMAPARSE_URL}/v2/parse", headers=headers, json=job, timeout=60))

    # 3. Wait until the job is done
    while True:
        result = check(httpx.get(f"{LLAMAPARSE_URL}/v2/parse/{reply['id']}", headers=headers,
                                 params={"expand": "markdown"}, timeout=60))
        if result["job"]["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
            break
        time.sleep(0.5)

    if result["job"]["status"] != "COMPLETED":
        raise SystemExit(result)
    return [page["markdown"] for page in result["markdown"]["pages"]]


def parse_with_mistral(pdf_bytes):
    # Azure takes the PDF inside the JSON, as a base64 data URL
    document_url = "data:application/pdf;base64," + base64.b64encode(pdf_bytes).decode()
    request = {"model": "mistral-ocr-4-0", "document": {"type": "document_url", "document_url": document_url}}
    reply = check(httpx.post(AZURE_MISTRAL_URL, headers={"api-key": os.environ["AZURE_API_KEY"]}, json=request,
                             timeout=300))
    return [page["markdown"] for page in reply["pages"]]


# Each one takes the PDF's bytes and gives back the markdown of each page
parsers = {
    "liteparse": parse_with_liteparse,
    "docling": parse_with_docling,
    "watsonx": parse_with_watsonx,
    "llamaparse": parse_with_llamaparse,
    "mistral-ocr-4-0": parse_with_mistral,
}
pdf_bytes = PDF_PATH.read_bytes()
NAME_WIDTH = max(len(name) for name in parsers) + 2


def time_it(parse):
    """Seconds to parse the PDF once, and the markdown of each page."""
    start = perf_counter()
    pages = parse(pdf_bytes)
    return perf_counter() - start, pages


def benchmark(name, parse):
    # The first parse may also wake the server up
    first, pages = time_it(parse)
    again = median([time_it(parse)[0] for _ in range(RUNS)])

    markdown = "\n\n".join(f"<!-- page {number} -->\n{page}" for number, page in enumerate(pages, start=1))
    (OUTPUT_DIR / f"{PDF_PATH.stem}_{name}.md").write_text(markdown)
    characters = sum(len(page) for page in pages)
    print(f"{name:<{NAME_WIDTH}}{len(pages):>6}{characters:>8}{first:>8.3f}{again:>8.3f}")


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"{PDF_PATH.name}. Seconds: first = one run, median = median of the next {RUNS}.\n")
print(f"{'service':<{NAME_WIDTH}}{'pages':>6}{'chars':>8}{'first':>8}{'median':>8}")
for name, parse in parsers.items():
    benchmark(name, parse)
