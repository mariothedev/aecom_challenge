"""Run the pipeline on the PDFs in data/ and check that it finds the known contradictions.

    python3 evaluate.py         # empty the database, upload every PDF in data/, then score
    python3 evaluate.py scan    # keep the stored statements, only look for contradictions again, then score
"""

import asyncio
import re
import sys
from time import perf_counter

import psycopg
from fastapi.testclient import TestClient

import main

DATA_DIR = main.BASE_DIR / "data"

# The contradictions in data/, found by reading the three PDFs by hand.
# A finding is one of them when one statement matches every pattern in "a" and the other every pattern in "b".
EXPECTED = [
    {"name": "Max dogs in D-01: 6 vs 8, then 5 and 4 by the addenda",
     "a": [r"dog", r"at most|max"], "b": [r"dog", r"at most|max"]},
    {"name": "Gate G-C01 height: 120 cm vs 90 cm",
     "a": [r"\b90 ?cm"], "b": [r"\b120 ?cm"]},
    {"name": "Dog bowl capacity: 2 litres vs 2 ml, fixed by Addendum 1",
     "a": [r"\b2 ?ml\b"], "b": [r"\b2 ?lit"]},
    {"name": "Cat bowl count: 2, changed to 3 by Addendum 2",
     "a": [r"cat|c-01", r"bowl", r"\b(2|two)\b"], "b": [r"cat|c-01", r"bowl", r"\b(3|three)\b"]},
    {"name": "Scratching posts: the Club Team vs owners supply them",
     "a": [r"scratching", r"club team must supply|supplied by the club team"], "b": [r"scratching", r"owner"]},
    {"name": "Cleaning: twice a day vs once a day",
     "a": [r"twice"], "b": [r"once each day|once a day|once daily"]},
    {"name": "Setup: programme ends 17 May (16 days) vs deadline 15 May (14 days)",
     "a": [r"17 may|16 days|16-day"], "b": [r"15 may|14 days"]},
    {"name": "Last day for questions: 20 April, changed to 25 April by Addendum 2",
     "a": [r"20 april"], "b": [r"25 april"]},
    {"name": "Setup checklist has no row for first-aid box FA-01",
     "a": [r"checklist", r"fa-01|first.aid"], "b": [r"checklist"]},
]

# These look like contradictions but are not. A finding that matches one is a false alarm.
NOT_CONTRADICTIONS = [
    {"name": "Cat bowl 500 ml vs 0.5 litres (the same amount)", "a": [r"500 ?ml"], "b": [r"0\.5 ?lit"]},
    {"name": "Puppy Room vs Dog Room (the same room)", "a": [r"puppy"], "b": [r"puppy|dog room"]},
    {"name": "Dog beds should be blue vs may be green (blue is only a preference)", "a": [r"blue"], "b": [r"green"]},
]


def flat(text):
    return re.sub(r"\s+", " ", text or "").strip().lower()


def has_all(patterns, text):
    return all(re.search(pattern, text) for pattern in patterns)


def find_match(items, text_a, text_b):
    """The first item this pair of statements is about, or None."""
    for item in items:
        if has_all(item["a"], text_a) and has_all(item["b"], text_b):
            return item
        if has_all(item["a"], text_b) and has_all(item["b"], text_a):
            return item
    return None


def empty_database():
    asyncio.run(main.create_table())
    with psycopg.connect(main.DATABASE_URL) as conn:
        conn.execute("DELETE FROM statements")  # also removes the contradictions (ON DELETE CASCADE)


def upload():
    files = [("files", (pdf.name, pdf.read_bytes(), "application/pdf")) for pdf in sorted(DATA_DIR.glob("*.pdf"))]
    with TestClient(main.app) as client:  # "with" runs the app's startup: create tables, warm up
        print(client.post("/upload", files=files).text)


async def scan():
    """The same steps /upload runs after the statements are stored."""
    # await main.cluster_statements()  # no clustering step any more
    await main.find_contradictions()


def load_findings():
    with psycopg.connect(main.DATABASE_URL) as conn:
        return conn.execute("""
            SELECT a.document, a.page, a.entity, a.text, b.document, b.page, b.entity, b.text, c.reason
            FROM contradictions c
            JOIN statements a ON a.id = c.statement_a
            JOIN statements b ON b.id = c.statement_b
            ORDER BY c.id""").fetchall()


def report(findings, seconds):
    found, false_alarms, others = [], 0, 0
    print(f"\n=== {len(findings)} finding(s) ===")
    for doc_a, page_a, entity_a, text_a, doc_b, page_b, entity_b, text_b, reason in findings:
        a, b = flat(f"{entity_a}: {text_a}"), flat(f"{entity_b}: {text_b}")
        expected = find_match(EXPECTED, a, b)
        trap = find_match(NOT_CONTRADICTIONS, a, b)
        if expected and expected["name"] in found:
            label = expected["name"] + " (again)"
        elif expected:
            label = expected["name"]
            found.append(expected["name"])
        elif trap:
            label = "FALSE ALARM: " + trap["name"]
            false_alarms += 1
        else:
            label = "OTHER (check by hand)"
            others += 1
        print(f"\n-> {label}")
        print(f"   {reason}")
        print(f"   - {doc_a[:2]} p{page_a} {entity_a}: {text_a}")
        print(f"   - {doc_b[:2]} p{page_b} {entity_b}: {text_b}")

    print("\n=== Score ===")
    print(f"Found:        {len(found)}/{len(EXPECTED)}")
    print(f"False alarms: {false_alarms}")
    print(f"Other:        {others}  (not in the lists above: read them and decide)")
    for expected in EXPECTED:
        if expected["name"] not in found:
            print(f"Missed:       {expected['name']}")
    print(f"Time:         {seconds:.3f} seconds")


if __name__ == "__main__":
    start = perf_counter()
    if sys.argv[1:] == ["scan"]:
        asyncio.run(scan())
    else:
        empty_database()
        upload()
    report(load_findings(), perf_counter() - start)


# ---------------------------------------------------------------------------------------------
# The old evaluate.py, kept for reference. It used an older API (/uploads, /extract, /supersede,
# /scan and a "chunks" table) and a minimal_data/ folder, none of which exist any more.
# ---------------------------------------------------------------------------------------------
# """Run the pipeline on minimal_data and check that it finds the known contradictions.
#
#     python3 evaluate.py            # upload, extract, supersede, scan
#     python3 evaluate.py scan       # start at a later step, reusing what is already stored
# """
#
# import re
# import sys
#
# import psycopg
# from fastapi.testclient import TestClient
#
# import main
#
# DATA_DIR = main.BASE_DIR / "minimal_data"
# STEPS = ["upload", "extract", "supersede", "scan"]
#
# # The contradictions planted in minimal_data. A finding is one of them when its
# # quotes contain every piece of text listed under "quotes_contain".
# EXPECTED = [
#     {"name": "Gate D-G02 height", "category": "numeric",
#      "quotes_contain": ["1,200", "900"]},
#     {"name": "Programme finish vs completion date", "category": "date_arithmetic",
#      "quotes_contain": ["17 May", "15 May"]},
#     {"name": "Dog Room limit, settled by the addenda", "category": "superseded_text",
#      "quotes_contain": ["6 dogs", "8 dogs"]},
# ]
#
#
# def flat(text):
#     return re.sub(r"\s+", " ", text).strip().lower()
#
#
# def matching_expected(finding):
#     """The expected contradiction this finding is about, or None."""
#     quotes = flat(" ".join(statement["quote"] for statement in finding["statements"]))
#     for expected in EXPECTED:
#         if all(flat(piece) in quotes for piece in expected["quotes_contain"]):
#             return expected
#     return None
#
#
# def run_pipeline(first_step):
#     client = TestClient(main.app)
#     steps = STEPS[STEPS.index(first_step):]
#
#     if "upload" in steps:
#         # Start from an empty database, so nothing from another pack is compared.
#         with psycopg.connect(main.DATABASE_URL) as conn:
#             main.ensure_schema(conn)
#             conn.execute("DELETE FROM chunks")
#         files = [("files", (pdf.name, pdf.read_bytes(), "application/pdf"))
#                  for pdf in sorted(DATA_DIR.glob("*.pdf"))]
#         print(client.post("/uploads", files=files).text)
#     if "extract" in steps:
#         print(client.post("/extract").text)
#     if "supersede" in steps:
#         print(client.post("/supersede").text)
#     return client.post("/scan").json()["findings"]
#
#
# def report(findings):
#     print(f"\n=== {len(findings)} finding(s) ===")
#     found, extra, right_category = [], 0, 0
#     for finding in findings:
#         expected = matching_expected(finding)
#         if expected is None or expected["name"] in found:
#             label = "EXTRA"
#             extra += 1
#         else:
#             found.append(expected["name"])
#             label = expected["name"]
#             if finding["category"] == expected["category"]:
#                 right_category += 1
#             else:
#                 label += f" (category should be {expected['category']})"
#         print(f"\n[{finding['category']} / {finding['severity']} / {finding['confidence']:.2f}] "
#               f"{finding['entity']} . {finding['attribute']}  ->  {label}")
#         print(f"    {finding['explanation']}")
#         for statement in finding["statements"]:
#             print(f"    - {statement['doc'][:2]} p{statement['page']} [{statement['status']}, "
#                   f"{statement['kind']}] {statement['quote']}")
#
#     print("\n=== Score ===")
#     print(f"Found:          {len(found)}/{len(EXPECTED)}")
#     print(f"Right category: {right_category}/{len(found)}")
#     print(f"Extra:          {extra}")
#     for expected in EXPECTED:
#         if expected["name"] not in found:
#             print(f"Missed:         {expected['name']}")
#
#
# if __name__ == "__main__":
#     first_step = sys.argv[1] if len(sys.argv) > 1 else STEPS[0]
#     if first_step not in STEPS:
#         sys.exit(f"First step must be one of: {', '.join(STEPS)}")
#     report(run_pipeline(first_step))
