
# This is a high-level overview of how the app works: 

# Ingestion flow

1. zip (pdfs) upload
2. PyMuPDF (text/page), pdfplumber (tables) - OCR as fallback (if no usable text to extract)
3. Chunking: split extracted content by sections, clauses and tables.
 - chunk_id, doc, page, section heading, text
Ex: 
```json
{
  "chunk_id": "chunk_001",
  "doc": "example_specification.pdf",
  "page": 3,
  "section_heading": "6 - Ventilation",
  "text": "6.1 Every occupied room must have ventilation.\n6.2 Main Hall ventilation must be 8 l/s per person."
},
{
  "chunk_id": "chunk_002",
  "doc": "example_specification.pdf",
  "page": 3,
  "section_heading": "6 - Ventilation",
  "text": "6.2 Main Hall ventilation must be 8 l/s per person."
}
```
4. Embedding model: embed chunks back into the table:
 - One row per chunk
```txt
| chunk_id | doc | page | section_heading | text | embedding |
|---|---|---|---|---|---|
| chunk_001 | example_specification.pdf | 3 | 6 - Ventilation | Main Hall ventilation must be… | [0.12, -0.34, …] |
```
-----------
At this point, we have the chunks and its vectorized versions in the db! Now, we have the LLM process it and apply it's findings. This LLM-processed content will 
-----------
5. Read each chunk’s text from the database and ask the LLM to turn its requirements into structured facts. 
ex:
from:
“Main Hall ventilation must be 8 l/s per person.”
and store in `facts` table:
```json
{
  "entity": "Main Hall",
  "attribute": "ventilation rate",
  "value": 8,
  "unit": "l/s per person",
  "source_chunk_id": "chunk_001",
  "quote": "Main Hall ventilation must be 8 l/s per person."
}
```
6. Normalize entity names. Give different names for the same thing one consistent name.
- a `canonical_entiry` column entry to facts table, ex:
```json
{
  "entity": "Main Hall",
  "canonical_entity": "main_hall",
  "attribute": "ventilation rate",
  "value": 8,
  "unit": "l/s per person",
  "source_chunk_id": "chunk_001",
  "quote": "Main Hall ventilation must be 8 l/s per person."
}
```
7. Mark facts that an addendum replaces in `facts` table

* An addendum is an official update to the tender documents.
* "active" here is the rule we follow, superceded means no longer active

Ex:
```
| Source | Ventilation rate | Status |
|---|---|---|
| Original specification | 12 l/s per person | superseded |
| Addendum 1 | 10 l/s per person | superseded |
| Addendum 2 | 8 l/s per person | active |
```
the Main Hall ventilation requirement to follow is 8 l/s per person with CO₂ demand control.

```json
{
  "fact_id": "fact_001",
  "entity": "Main Hall",
  "canonical_entity": "main_hall",
  "attribute": "ventilation rate",
  "value": 8,
  "unit": "l/s per person",
  "source_chunk_id": "chunk_001",
  "quote": "Main Hall ventilation must be 8 l/s per person.",
  "status": "active", // superceded|active
  "superceded_by_fact_id": "fact_002", // null
}
```
*******************************
*******************************
User clicks scan...
*******************************
*******************************

1. Query facts that talk about the same thing and same property in the `facts` table.

Ex:
main_hall + ventilation_rate
```
| Fact | Source | Value | Status |
|---|---|---|---|
| fact_003 | Addendum 2 | 8 l/s per person | active |
| fact_020 | Another specification | 12 l/s per person | active |
```
These facts belong together because both describe Main Hall ventilation.

2. Python code identifies conflicts in the `facts` table and creates a candidate pair.

Ex:
```json
{
  "fact_a_id": "fact_003",
  "fact_b_id": "fact_020"
}
```
from:
```
| Fact | Entity | Attribute | Value |
|---|---|---|---|
| fact_003 | main_hall | ventilation_rate | 8 l/s per person |
| fact_020 | main_hall | ventilation_rate | 12 l/s per person |
```

3. At this point, the LLM judges whether it's a real contradiction.

Ex:
```json
{
  "is_conflict": true,
  "category": "numeric",
  "severity": "high",
  "confidence": 0.91,
  "reason": "Both specify different mandatory design rates for the same room under the same conditions."
}
```

4.  verify quotes against the source

The python code checks that each quote returned by the LLM actually exists in it's referenced source.

EX:  

```txt
LLM quote:   “The ventilation rate shall be 8 l/s per person.”
Source text: “The ventilation rate shall be 8 l/s per person.”

→ Match: quote verified.
```

5. 

Your Python code sorts them by:

1. **Severity:** high → medium → low.
2. **Confidence:** highest first within each severity level.

For example:

| Finding | Severity | Confidence |
|---|---|---|
| Conflicting fire ratings | High | 0.95 |
| Conflicting electrical loads | High | 0.88 |
| Conflicting finish colours | Low | 0.97 |

The colour finding stays below the high-severity findings, despite its higher confidence.

**No new LLM call here—just sorting the judgments from Step 11.**


**Step 14: save the verified findings in a `findings` table.**

Your Python code inserts each finding, including:

```json
{
  "finding_id": "finding_001",
  "fact_a_id": "fact_003",
  "fact_b_id": "fact_020",
  "category": "numeric",
  "severity": "high",
  "confidence": 0.91,
  "explanation": "Different mandatory ventilation rates for the same room.",
  "quote_a": "...verified source quote...",
  "quote_b": "...verified source quote..."
}
```

The fact IDs link back to the chunks, documents, pages, and headings.

Your UI reads this table and displays the findings **ordered by severity, then confidence**. Database rows have no guaranteed order, so apply that sorting when you query them.

**That completes the scan flow.** Next comes answering questions.



# Design notes

## How are contradictions compared


## How to keep heading, table and entity context
