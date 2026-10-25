"""Run the pipeline on minimal_data and check that it finds the known contradictions.

    python3 evaluate.py            # upload, extract, supersede, scan
    python3 evaluate.py scan       # start at a later step, reusing what is already stored
"""

import re
import sys

import psycopg
from fastapi.testclient import TestClient

import main

DATA_DIR = main.BASE_DIR / "minimal_data"
STEPS = ["upload", "extract", "supersede", "scan"]

# The contradictions planted in minimal_data. A finding is one of them when its
# quotes contain every piece of text listed under "quotes_contain".
EXPECTED = [
    {"name": "Gate D-G02 height", "category": "numeric",
     "quotes_contain": ["1,200", "900"]},
    {"name": "Programme finish vs completion date", "category": "date_arithmetic",
     "quotes_contain": ["17 May", "15 May"]},
    {"name": "Dog Room limit, settled by the addenda", "category": "superseded_text",
     "quotes_contain": ["6 dogs", "8 dogs"]},
]


def flat(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def matching_expected(finding):
    """The expected contradiction this finding is about, or None."""
    quotes = flat(" ".join(statement["quote"] for statement in finding["statements"]))
    for expected in EXPECTED:
        if all(flat(piece) in quotes for piece in expected["quotes_contain"]):
            return expected
    return None


def run_pipeline(first_step):
    client = TestClient(main.app)
    steps = STEPS[STEPS.index(first_step):]

    if "upload" in steps:
        # Start from an empty database, so nothing from another pack is compared.
        with psycopg.connect(main.DATABASE_URL) as conn:
            main.ensure_schema(conn)
            conn.execute("DELETE FROM chunks")
        files = [("files", (pdf.name, pdf.read_bytes(), "application/pdf"))
                 for pdf in sorted(DATA_DIR.glob("*.pdf"))]
        print(client.post("/uploads", files=files).text)
    if "extract" in steps:
        print(client.post("/extract").text)
    if "supersede" in steps:
        print(client.post("/supersede").text)
    return client.post("/scan").json()["findings"]


def report(findings):
    print(f"\n=== {len(findings)} finding(s) ===")
    found, extra, right_category = [], 0, 0
    for finding in findings:
        expected = matching_expected(finding)
        if expected is None or expected["name"] in found:
            label = "EXTRA"
            extra += 1
        else:
            found.append(expected["name"])
            label = expected["name"]
            if finding["category"] == expected["category"]:
                right_category += 1
            else:
                label += f" (category should be {expected['category']})"
        print(f"\n[{finding['category']} / {finding['severity']} / {finding['confidence']:.2f}] "
              f"{finding['entity']} . {finding['attribute']}  ->  {label}")
        print(f"    {finding['explanation']}")
        for statement in finding["statements"]:
            print(f"    - {statement['doc'][:2]} p{statement['page']} [{statement['status']}, "
                  f"{statement['kind']}] {statement['quote']}")

    print("\n=== Score ===")
    print(f"Found:          {len(found)}/{len(EXPECTED)}")
    print(f"Right category: {right_category}/{len(found)}")
    print(f"Extra:          {extra}")
    for expected in EXPECTED:
        if expected["name"] not in found:
            print(f"Missed:         {expected['name']}")


if __name__ == "__main__":
    first_step = sys.argv[1] if len(sys.argv) > 1 else STEPS[0]
    if first_step not in STEPS:
        sys.exit(f"First step must be one of: {', '.join(STEPS)}")
    report(run_pipeline(first_step))
