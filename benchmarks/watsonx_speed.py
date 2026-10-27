"""Time how long watsonx.ai text extraction takes to turn a PDF into markdown.

    python3 watsonx_speed.py

Needs WATSONX_API_KEY in .env and a task credential in your IBM account (one-time setup).
"""

import os
import time
from pathlib import Path
from time import perf_counter

import httpx
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "01_Club_Rules_and_Requirements.pdf"
OUTPUT_PATH = BASE_DIR / "markdown" / "01_Club_Rules_and_Requirements_watsonx.md"

load_dotenv(BASE_DIR / ".env", override=True)
API_KEY = os.environ["WATSONX_API_KEY"]
PROJECT_ID = "9e48aa77-4089-41a5-aad7-d6aae062ce85"
WATSONX_URL = "https://eu-de.ml.cloud.ibm.com"
# The project's own storage bucket
BUCKET_URL = "https://s3.eu-de.cloud-object-storage.appdomain.cloud/mariossandbox-donotdelete-pr-b18vpwbnxsqqz5"
VERSION = "2024-05-01"

INPUT_FILE = "speed_test/" + PDF_PATH.name
RESULT_FILE = "speed_test/" + OUTPUT_PATH.name


def get_token():
    response = httpx.post(
        "https://iam.cloud.ibm.com/identity/token",
        data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": API_KEY},
    )
    response.raise_for_status()
    return response.json()["access_token"]


headers = {"Authorization": f"Bearer {get_token()}"}
project = {"project_id": PROJECT_ID}

start = perf_counter()

# 1. Put the PDF in the project's storage
response = httpx.put(f"{BUCKET_URL}/{INPUT_FILE}", headers=headers | {"Content-Type": "application/pdf"},
                     content=PDF_PATH.read_bytes(), timeout=60)
response.raise_for_status()

# 2. Start the extraction job
job = {
    "project_id": PROJECT_ID,
    "document_reference": {"type": "container", "location": {"path": INPUT_FILE}},
    "results_reference": {"type": "container", "location": {"path": RESULT_FILE}},
    "parameters": {"requested_outputs": ["md"], "mode": "standard"},
}
response = httpx.post(f"{WATSONX_URL}/ml/v1/text/extractions", params={"version": VERSION},
                      headers=headers, json=job, timeout=60)
if response.is_error:
    raise SystemExit(response.text)
job_id = response.json()["metadata"]["id"]

# 3. Wait until the job is done
while True:
    response = httpx.get(f"{WATSONX_URL}/ml/v1/text/extractions/{job_id}",
                         params=project | {"version": VERSION}, headers=headers, timeout=60)
    response.raise_for_status()
    results = response.json()["entity"]["results"]
    if results["status"] in ("completed", "failed"):
        break
    time.sleep(1)

if results["status"] == "failed":
    raise SystemExit(results)

# 4. Download the markdown
response = httpx.get(f"{BUCKET_URL}/{RESULT_FILE}", headers=headers, timeout=60)
response.raise_for_status()
markdown = response.text
print(f"watsonx: {perf_counter() - start:.3f} seconds, {len(markdown)} characters")
print(results)

OUTPUT_PATH.write_text(markdown)
