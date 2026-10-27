"""Time how long Google Document AI (Enterprise Document OCR) takes to read a PDF.

    pip install google-cloud-documentai
    python3 google_speed.py

Needs google-key.json (a service account key with the Document AI API User role) in this folder.
"""

from pathlib import Path
from time import perf_counter

from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from google.oauth2 import service_account

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "01_Club_Rules_and_Requirements.pdf"
OUTPUT_PATH = BASE_DIR / "markdown" / "01_Club_Rules_and_Requirements_google.txt"  # OCR gives plain text
KEY_PATH = BASE_DIR / "google-key.json"

PROJECT_ID = "89779903351"
LOCATION = "europe-west2"
PROCESSOR_ID = "196c72bfa4700bf6"
PROCESSOR_VERSION = "pretrained-ocr-v2.0-2023-06-02"

credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
client = documentai.DocumentProcessorServiceClient(
    credentials=credentials,
    client_options=ClientOptions(api_endpoint=f"{LOCATION}-documentai.googleapis.com"),
)
name = client.processor_version_path(PROJECT_ID, LOCATION, PROCESSOR_ID, PROCESSOR_VERSION)

start = perf_counter()
result = client.process_document(request=documentai.ProcessRequest(
    name=name,
    raw_document=documentai.RawDocument(content=PDF_PATH.read_bytes(), mime_type="application/pdf"),
))
text = result.document.text
print(f"google: {perf_counter() - start:.3f} seconds, {len(text)} characters")

OUTPUT_PATH.write_text(text)
