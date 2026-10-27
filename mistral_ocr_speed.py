"""Time how long Mistral OCR on Vertex AI takes to turn a PDF into markdown.

    python3 mistral_ocr_speed.py

Needs google-key.json (a service account key with the Vertex AI User role) in this folder.
"""

import base64
from pathlib import Path
from time import perf_counter

import httpx
from google.auth.transport.requests import Request
from google.oauth2 import service_account

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "01_Club_Rules_and_Requirements.pdf"
OUTPUT_PATH = BASE_DIR / "markdown" / "01_Club_Rules_and_Requirements_mistral_ocr.md"
KEY_PATH = BASE_DIR / "google-key.json"

LOCATION = "us-central1"  # the only region Model Garden lists for this model
MODEL = "mistral-ocr-2505"

credentials = service_account.Credentials.from_service_account_file(
    KEY_PATH, scopes=["https://www.googleapis.com/auth/cloud-platform"])
credentials.refresh(Request())
headers = {"Authorization": f"Bearer {credentials.token}"}

# The project the service key belongs to
OCR_URL = (f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{credentials.project_id}"
           f"/locations/{LOCATION}/publishers/mistralai/models/{MODEL}:rawPredict")

document_url = "data:application/pdf;base64," + base64.b64encode(PDF_PATH.read_bytes()).decode()
body = {"model": MODEL, "document": {"type": "document_url", "document_url": document_url}}

start = perf_counter()
response = httpx.post(OCR_URL, headers=headers, json=body, timeout=300)
if response.is_error:
    raise SystemExit(response.text)
pages = response.json()["pages"]
markdown = "\n\n".join(f"<!-- page {page['index'] + 1} -->\n{page['markdown']}" for page in pages)
print(f"mistral ocr: {perf_counter() - start:.3f} seconds, {len(pages)} pages, {len(markdown)} characters")

OUTPUT_PATH.write_text(markdown)
