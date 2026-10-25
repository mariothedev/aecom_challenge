"""Step 2: upload tender PDFs, extract structured chunks, store them in PostgreSQL."""

import io
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import pdfplumber
import psycopg
import pymupdf
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
# Optional local settings; existing host environment variables take precedence.
load_dotenv(BASE_DIR / ".env", override=False)
DATABASE_URL = os.environ["DATABASE_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_API_BASE = "https://api.openai.com/v1"

HEADER_Y = 50  # running headers sit above this point on the page
FOOTER_Y = 790  # page numbers sit below this point
MAX_HEADING_CHARS = 120  # a heading is always a short line
MAX_CHUNK_CHARS = 1800  # split longer sections so chunks stay retrievable

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1024  # keep the existing pgvector column size
EMBEDDING_BATCH = 64

CHAT_MODEL = "gpt-6.1-sol"
EXTRACT_CONCURRENCY = 8  # chunks read in parallel, so a full pack does not take minutes
CANDIDATE_LIMIT = 30  # statements from nearby chunks an amendment is compared against


app = FastAPI(title="Tender contradiction detection")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


# --------------------------------------------------------------------------
# Reading the PDF
# --------------------------------------------------------------------------

def is_bold(span):
    return bool(span["flags"] & 16) or "Bold" in span["font"]


def line_text(line):
    return re.sub(r"\s+", " ", "".join(s["text"] for s in line["spans"])).strip()


def body_blocks(page):
    """Text blocks of one page, without the running header and footer."""
    for block in page.get_text("dict")["blocks"]:
        spans = [s for line in block.get("lines", []) for s in line["spans"] if s["text"].strip()]
        if not spans:
            continue
        top, bottom = block["bbox"][1], block["bbox"][3]
        if top < HEADER_Y or bottom > FOOTER_Y:
            continue
        yield block, spans


def body_size(doc):
    """The most used font size of plain text, i.e. the size of normal body copy."""
    used = {}
    for page in doc.pages(1, doc.page_count):  # skip the cover, it is all large type
        for _, spans in body_blocks(page):
            for span in spans:
                if not is_bold(span):
                    size = round(span["size"], 1)
                    used[size] = used.get(size, 0) + len(span["text"].strip())
    return max(used, key=used.get) if used else 0


def heading_size(doc, body):
    """The font size of top level headings: the largest size used by bold short lines."""
    sizes = []
    for page in doc.pages(1, doc.page_count):
        for block, spans in body_blocks(page):
            size = round(max(s["size"] for s in spans), 1)
            text = " ".join(line_text(line) for line in block["lines"]).strip()
            if is_bold(spans[0]) and size >= body and len(text) <= MAX_HEADING_CHARS:
                sizes.append(size)
    return max(sizes) if sizes else 0


def document_title(doc, filename):
    """The document's own title, printed in the largest type on the cover page."""
    lines = []
    for block in doc[0].get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = line_text(line)
            if text:
                lines.append((round(max(s["size"] for s in line["spans"]), 1), text))
    if not lines:
        return Path(filename).stem
    # The cover prints the project name first, then "Document NN", then the title itself.
    start = 0
    for index, (_, text) in enumerate(lines):
        if re.match(r"^Document\s+\d+", text):
            start = index + 1
    biggest = max(size for size, _ in lines)
    title = [text for size, text in lines[start:] if size == biggest]
    return " ".join(title) if title else Path(filename).stem


def page_pieces(page, plumber_page):
    """Everything on a page in reading order, as (text, font size, is bold, is table)."""
    tables = []
    for table in plumber_page.find_tables():
        rows = []
        for row in table.extract():
            cells = [re.sub(r"\s+", " ", cell or "").strip() for cell in row]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            tables.append((table.bbox, "\n".join(rows)))

    pieces = []
    for block, spans in body_blocks(page):
        left, top, right, bottom = block["bbox"]
        middle_x, middle_y = (left + right) / 2, (top + bottom) / 2
        # pdfplumber already captured this text as a table row, so do not repeat it.
        if any(x0 <= middle_x <= x1 and y0 <= middle_y <= y1 for (x0, y0, x1, y1), _ in tables):
            continue
        text = " ".join(line_text(line) for line in block["lines"]).strip()
        if text:
            pieces.append((top, text, round(max(s["size"] for s in spans), 1), is_bold(spans[0]), False))
    for (_, top, _, _), text in tables:
        pieces.append((top, text, 0, False, True))

    pieces.sort(key=lambda piece: piece[0])
    return [(text, size, bold, is_table) for _, text, size, bold, is_table in pieces]


# --------------------------------------------------------------------------
# Turning a document into chunks
# --------------------------------------------------------------------------

def heading_depth(text):
    """How deep a heading sits among headings printed in the same size.

    '5. Doors' is 1, '5.2 Door schedule' and 'Item 1.1 - ...' are 2. A heading that starts
    with a code, like 'TQ-01' or 'A-101', names one item of the section above it.
    """
    if re.match(r"^[A-Z][A-Z0-9]*-\d", text):
        return 9
    number = re.match(r"^(?:[A-Za-z]+\s+)?(\d+(?:\.\d+)*)", text)
    return number.group(1).count(".") + 1 if number else 1


def split_text(text):
    """Cut long prose at sentence ends."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if current and len(current) + 1 + len(sentence) > MAX_CHUNK_CHARS:
            pieces.append(current)
            current = ""
        current = f"{current} {sentence}" if current else sentence
    if current:
        pieces.append(current)
    return pieces


def split_table(text):
    """Cut a long table between rows. Every piece starts with the header row, so a row
    like 'D-105 | ... | 2,100' still says which column is the height."""
    header, *rows = text.split("\n")
    pieces, current = [], header
    for row in rows:
        if current != header and len(current) + 1 + len(row) > MAX_CHUNK_CHARS:
            pieces.append(current)
            current = header
        current += "\n" + row
    pieces.append(current)
    return pieces


def pack(items):
    """Join a section's (page, text, is table) items into chunks of at most MAX_CHUNK_CHARS.
    Returns (page, text) pairs, where page is the page the chunk starts on."""
    parts = []
    for page, text, is_table in items:
        for part in split_table(text) if is_table else split_text(text):
            parts.append((page, part))

    packed, current, current_page = [], "", 0
    for page, part in parts:
        if current and len(current) + 1 + len(part) > MAX_CHUNK_CHARS:
            packed.append((current_page, current))
            current = ""
        if current:
            current += "\n" + part
        else:
            current, current_page = part, page
    if current:
        packed.append((current_page, current))
    return packed


def make_chunks(content, filename):
    """Read one PDF and return its chunks, one per section (split if very long)."""
    doc = pymupdf.open(stream=content, filetype="pdf")
    plumber = pdfplumber.open(io.BytesIO(content))

    title = document_title(doc, filename)
    body = body_size(doc)
    top_size = heading_size(doc, body)

    chunks = []
    headings = []  # (level, text) of each heading above the current text, outermost first
    buffer = []  # (page, text, is table) of the current section

    def flush():
        parents = " > ".join(text for _, text in headings[:-1])
        heading = headings[-1][1] if headings else ""
        for page, text in pack(buffer):
            chunks.append({
                "doc": filename,
                "page": page,
                "page_title": title,
                "section_parent": parents,
                "section_heading": heading,
                "section_content": text,
            })
        buffer.clear()

    for index in range(doc.page_count):
        for text, size, bold, is_table in page_pieces(doc[index], plumber.pages[index]):
            if bold and size >= body and len(text) <= MAX_HEADING_CHARS:
                flush()
                # Bigger type is higher up. The cover's huge title counts as a top heading,
                # so it does not sit above every section of the document.
                level = (-min(size, top_size), heading_depth(text))
                while headings and headings[-1][0] >= level:
                    headings.pop()
                headings.append((level, text))
                continue
            buffer.append((index + 1, text, is_table))
    flush()

    doc.close()
    plumber.close()
    return chunks


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------

def embed(texts):
    """Turn texts into vectors, a batch at a time."""
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
        base_url=OPENAI_API_BASE,
        api_key=OPENAI_API_KEY,
        chunk_size=EMBEDDING_BATCH,
        max_retries=3,
    )
    return embeddings.embed_documents(texts)


def embedding_text(chunk):
    """What we actually embed: the content plus the headings that give it context.

    A table row like "FZ-01 | 4 | kW" means little on its own; prefixing its document
    and section makes it findable by questions phrased either way.
    """
    headings = [chunk["page_title"], chunk["section_parent"], chunk["section_heading"]]
    return " > ".join(h for h in headings if h) + "\n" + chunk["section_content"]


def to_vector(values):
    """pgvector reads a vector from a literal like '[0.1,0.2]'."""
    return "[" + ",".join(map(str, values)) + "]"


# --------------------------------------------------------------------------
# Reading statements out of a chunk
# --------------------------------------------------------------------------

class Statement(BaseModel):
    """One thing the pack states about one entity."""

    entity: str = Field(
        description="The thing the value describes, as the document names it: a room, door, "
                    "piece of equipment, the Works and so on. Never the document, drawing or "
                    "table the value is printed in."
    )
    equipment_tag: str | None = Field(
        default=None, description="Equipment, door, room or drawing tag such as FZ-01, if one is given."
    )
    attribute: str = Field(
        description="The property being fixed, in snake_case, e.g. electrical_load, fire_rating, "
                    "clear_height, completion_date."
    )
    value: str | None = Field(
        default=None, description="The value alone, without its unit, e.g. '400', '12', 'stainless steel'."
    )
    unit: str | None = Field(default=None, description="The unit alone, e.g. 'mm', 'kW', 'months'.")
    quote: str = Field(description="The shortest run of words from the text that proves this, copied exactly.")
    kind: Literal["requirement", "definition", "information", "document_control"] = Field(
        description="requirement: fixes something about the Works; definition: says what a term, "
                    "name or code means; information: marked as not binding; document_control: "
                    "about the documents themselves rather than the Works."
    )


class StatementList(BaseModel):
    statements: list[Statement]


EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You read tender documents and record what they require.\n"
     "List every statement in the text that fixes a value, a quantity, a date, a "
     "responsibility or a rule. Split each one so that a single statement covers a "
     "single attribute of a single entity.\n"
     "Rules:\n"
     "- Quote exactly. Copy the words from the text; never paraphrase or correct them.\n"
     "- Keep value and unit apart: '120' and 'cm', not '120 cm'.\n"
     "- Use the same snake_case attribute name for the same property wherever it appears, "
     "so the same property can be compared across documents.\n"
     "- Record what the text says even when it looks wrong or disagrees with another document.\n"
     "- Skip background prose, introductions and anything that fixes nothing.\n"
     "- Give every statement a kind:\n"
     "  requirement: anything fixed about the Works, such as a value, quantity, date, "
     "duration, rating, responsibility or programme activity;\n"
     "  definition: what a term, name, code or abbreviation means;\n"
     "  information: text that says itself it is not binding, such as an answer to a "
     "tender query or a note for information only;\n"
     "  document_control: facts about the documents rather than the Works, such as "
     "revisions, issue dates, authors, references, lists of documents, order of "
     "precedence, acknowledgements and notes saying what an addendum replaces.\n"
     "Return an empty list if the text states no requirement."),
    ("human",
     "Document: {doc}\nPage: {page}\nSection: {section}\n\n{content}"),
])


def extract_chain():
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        base_url=OPENAI_API_BASE,
        api_key=OPENAI_API_KEY,
        temperature=0,
        max_retries=3,
    )
    return EXTRACT_PROMPT | llm.with_structured_output(StatementList, method="json_schema")


def extract_statements(chunks):
    """Ask the model what each chunk requires. Returns (chunk_id, Statement) pairs."""
    chain = extract_chain()
    inputs = [
        {
            "doc": chunk["doc"],
            "page": chunk["page"],
            "section": " > ".join(
                h for h in (chunk["page_title"], chunk["section_parent"], chunk["section_heading"]) if h
            ),
            "content": chunk["section_content"],
        }
        for chunk in chunks
    ]
    results = chain.batch(
        inputs,
        config={"max_concurrency": EXTRACT_CONCURRENCY},
        return_exceptions=True,
    )

    found = []
    for chunk, result in zip(chunks, results):
        if isinstance(result, Exception):
            print(f"!! {chunk['chunk_id']} failed: {result}")
            continue
        for statement in result.statements:
            found.append((chunk, statement))
    return found


def quote_matches(quote, content):
    """True when the quote really is in the chunk, ignoring how whitespace fell."""
    flat = lambda text: re.sub(r"\s+", " ", text).strip().lower()
    return flat(quote) in flat(content)


# --------------------------------------------------------------------------
# Making values comparable
# --------------------------------------------------------------------------

# Each unit, lower-cased, with the base unit it becomes and the factor to get there.
UNITS = {
    "mm": ("mm", 1), "cm": ("mm", 10), "m": ("mm", 1000), "km": ("mm", 1_000_000),
    "m2": ("m2", 1), "m²": ("m2", 1),
    "ml": ("l", 0.001), "l": ("l", 1), "litre": ("l", 1), "litres": ("l", 1),
    "m3": ("l", 1000), "m³": ("l", 1000),
    "l/s": ("l/s", 1), "m3/h": ("l/s", 1 / 3.6), "m³/h": ("l/s", 1 / 3.6), "m3/s": ("l/s", 1000),
    "w": ("kw", 0.001), "kw": ("kw", 1), "mw": ("kw", 1000),
    "a": ("a", 1), "v": ("v", 1), "kv": ("v", 1000),
    "g": ("kg", 0.001), "kg": ("kg", 1), "t": ("kg", 1000), "tonne": ("kg", 1000), "tonnes": ("kg", 1000),
    "pa": ("pa", 1), "kpa": ("pa", 1000),
    "min": ("min", 1), "h": ("min", 60), "hour": ("min", 60), "hours": ("min", 60),
    "day": ("days", 1), "days": ("days", 1), "week": ("days", 7), "weeks": ("days", 7),
    "month": ("months", 1), "months": ("months", 1), "year": ("months", 12), "years": ("months", 12),
}


def parse_number(text):
    """'1,200' and '1 200' are 1200, '6.0' is 6, '1,5' is 1.5. None if text is not one plain number."""
    text = text.strip()
    if re.fullmatch(r"-?\d{1,3}(?:[ ,]\d{3})+(?:\.\d+)?", text):
        return float(re.sub(r"[ ,]", "", text))
    if re.fullmatch(r"-?\d+(?:[.,]\d+)?", text):
        return float(text.replace(",", "."))
    return None


def parse_date(text):
    """'15 May 2027', '2027-05-15' and '15.05.2027' all become '2027-05-15'. None if not a date."""
    text = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text.strip())
    for pattern in ("%d %B %Y", "%d %b %Y", "%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, pattern).date().isoformat()
        except ValueError:
            pass
    return None


def normalise(value, unit):
    """One comparable form of a value, so that equal values look equal:
    '1,200' mm -> '1200 mm', '1.2' m -> '1200 mm', '15 May 2027' -> '2027-05-15'.
    Anything that is not a plain number or a date is only lower-cased."""
    text = " ".join(part.strip() for part in (value, unit) if part and part.strip())
    if not text:
        return None
    date = parse_date(value or "")
    if date:
        return date
    # A number, then maybe a unit: '1,200 mm', '6 dogs', '6.0 m (double height)'.
    match = re.fullmatch(r"(-?\d[\d ,.]*?)\s*([^\d\s][^\d]*)?", text)
    number = parse_number(match.group(1)) if match else None
    if number is None:
        return re.sub(r"\s+", " ", text).lower()
    unit_text = (match.group(2) or "").strip().lower()
    first_word = unit_text.split(" ")[0]
    if first_word in UNITS:
        base, factor = UNITS[first_word]
        number, unit_text = number * factor, base
    number_text = f"{number:.6f}".rstrip("0").rstrip(".")
    return f"{number_text} {unit_text}".strip()


# --------------------------------------------------------------------------
# Facts worked out from a programme
# --------------------------------------------------------------------------

def programme_rows(chunk):
    """(start, finish, row) for every activity of a programme table in this chunk.
    A programme table is one whose header has a start column and a finish column."""
    rows, start_column, finish_column = [], None, None
    for line in chunk["section_content"].split("\n"):
        cells = [cell.strip() for cell in line.split(" | ")]
        starts = [i for i, cell in enumerate(cells) if re.search(r"\bstart", cell, re.I)]
        finishes = [i for i, cell in enumerate(cells) if re.search(r"\b(finish|end|completion)", cell, re.I)]
        if len(cells) > 1 and starts and finishes and not any(parse_date(cell) for cell in cells):
            start_column, finish_column = starts[0], finishes[0]  # a header row
            continue
        if start_column is None or len(cells) <= max(start_column, finish_column):
            continue
        start, finish = parse_date(cells[start_column]), parse_date(cells[finish_column])
        if start and finish:
            rows.append((start, finish, line))
    return rows


def programme_facts(chunks):
    """What each programme adds up to: the day it finishes and how long it runs, as
    statements about the Works. Compared with the contract, they show a programme that
    overruns the completion date, however its activities are named.
    Returns (chunk, attribute, value, unit, quote) for each fact."""
    programmes = {}
    for chunk in chunks:
        for start, finish, row in programme_rows(chunk):
            section = (chunk["doc"], chunk["section_parent"], chunk["section_heading"])
            programmes.setdefault(section, []).append((start, finish, row, chunk))

    facts = []
    for rows in programmes.values():
        first = min(rows, key=lambda row: row[0])
        last = max(rows, key=lambda row: row[1])
        days = (date.fromisoformat(last[1]) - date.fromisoformat(first[0])).days
        months = f"{days / 30.44:.1f}".rstrip("0").rstrip(".")
        facts.append((last[3], "completion_date", last[1], None, last[2]))
        quote = last[2] if first is last else f"{first[2]} ... {last[2]}"
        facts.append((last[3], "duration", months, "months", quote))
    return facts


# --------------------------------------------------------------------------
# Giving the same thing the same name
# --------------------------------------------------------------------------

class NameGroup(BaseModel):
    name: str = Field(description="The canonical name shared by everything in the group.")
    ids: list[str] = Field(description="Ids from the list given.")


class NameGroups(BaseModel):
    groups: list[NameGroup]


ENTITY_NAMES_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "The documents of a tender pack were read one section at a time, so the same thing "
     "was often named in different ways. Below is every name that was used, with its tag "
     "and some of the properties stated about it. Put the names into groups, one group per "
     "thing, and give each group the canonical name of the thing.\n"
     "Rules:\n"
     "- Names that refer to the same thing go in one group. Names that refer to different "
     "things go in different groups: a room, a door of that room and the equipment in it are "
     "three different things.\n"
     "- If a name or tag holds a code of letters and digits, such as a room, door or "
     "equipment number, the canonical name is that code and a short plain name, e.g. "
     "'FZ-01 Freezer'. Otherwise it is a short plain name, e.g. 'Water bowl'.\n"
     "- The project as a whole is one thing, whatever a document calls it: name it 'Works'. "
     "Its commencement date, completion date and overall duration belong to it.\n"
     "- Never make a catch-all group such as 'Document' or 'General'. Parties such as the "
     "Employer, terms being defined, rules about the documents and each part of the Works "
     "are different things, each in its own group.\n"
     "- Every id must be in exactly one group."),
    ("human", "{names}"),
])

ATTRIBUTE_NAMES_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "The documents of a tender pack were read one section at a time, so the same property "
     "of a thing was often named in different ways. Below is every property name used for "
     "one thing, with the values given for it. Put the names into groups, one group per "
     "property, and give each group a snake_case name.\n"
     "Two names are the same property when a reviewer would check that their values agree, "
     "e.g. 'fire_rating' and 'fire_resistance' of one door. Judge by meaning, not by value: "
     "documents often disagree, and those disagreements are what we are looking for.\n"
     "Keep apart properties that can rightly hold different values, e.g. 'width' and "
     "'height', or 'start_date' and 'completion_date'.\n"
     "Every id must be in exactly one group."),
    ("human", "Thing: {thing}\n\n{names}"),
])


def names_chain(prompt):
    llm = ChatOpenAI(model=CHAT_MODEL, base_url=OPENAI_API_BASE,
                     api_key=OPENAI_API_KEY, temperature=0, max_retries=3)
    return prompt | llm.with_structured_output(NameGroups, method="json_schema")


def snake_case(text):
    return re.sub(r"\W+", "_", text.strip().lower()).strip("_")


def codes(text):
    """Tags such as 'FZ-01', 'D-G02' or 'L00-001' found in a text."""
    return {code for code in re.findall(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+\b", text)
            if re.search(r"\d", code)}


def apply_names(answer, prefix, keys, names):
    """Store each group's name in names[key] for every key the group holds, found by its id."""
    for group in answer.groups:
        for item_id in group.ids:
            if re.fullmatch(prefix + r"\d+", item_id) and int(item_id[1:]) < len(keys):
                names[keys[int(item_id[1:])]] = group.name.strip()


def name_consistently(conn):
    """Give every statement a canonical entity and attribute name, so the same property
    of the same thing can be found in every document that mentions it."""
    rows = conn.execute(
        "SELECT statement_id, entity, equipment_tag, attribute, normal_value FROM statements"
    ).fetchall()

    # Things first: every name used, with a few of its properties so the model can tell
    # what it is. A name the model skips keeps its own spelling.
    properties_of = {}
    for _, entity, tag, attribute, _ in rows:
        properties_of.setdefault((entity, tag or ""), []).append(attribute)
    entity_keys = list(properties_of)
    lines = []
    for index, (entity, tag) in enumerate(entity_keys):
        properties = ", ".join(list(dict.fromkeys(properties_of[(entity, tag)]))[:3])
        lines.append(f"e{index}: {entity}" + (f" [tag {tag}]" if tag else "") + f" - {properties}")
    canonical_entity = {key: key[0] for key in entity_keys}
    answer = names_chain(ENTITY_NAMES_PROMPT).invoke({"names": "\n".join(lines)})
    apply_names(answer, "e", entity_keys, canonical_entity)
    # A tag is the surest sign of identity: a name never joins a thing tagged differently,
    # so 'L00-002' can not end up inside 'L00-001 Dog Room'.
    # Such a name goes to the one thing that carries its tag, or keeps its own spelling.
    for (entity, tag), thing in canonical_entity.items():
        own, given = codes(f"{entity} {tag}"), codes(thing)
        if own and given and not own & given:
            same_tag = {other for other in canonical_entity.values() if codes(other) & own}
            canonical_entity[(entity, tag)] = same_tag.pop() if len(same_tag) == 1 else entity

    # Then the properties of each thing, named with the thing in view, so a door's
    # 'height' and 'gate_height' can meet. A thing with one property has nothing to merge.
    values_of = {}
    for _, entity, tag, attribute, value in rows:
        thing = canonical_entity[(entity, tag or "")]
        values_of.setdefault(thing, {}).setdefault(attribute, []).append(value or "(none)")
    things = [thing for thing in values_of if len(values_of[thing]) > 1]
    inputs = []
    for thing in things:
        lines = [f"a{index}: {attribute} = {' / '.join(list(dict.fromkeys(values))[:3])}"
                 for index, (attribute, values) in enumerate(values_of[thing].items())]
        inputs.append({"thing": thing, "names": "\n".join(lines)})
    answers = names_chain(ATTRIBUTE_NAMES_PROMPT).batch(
        inputs, config={"max_concurrency": EXTRACT_CONCURRENCY}, return_exceptions=True
    )
    canonical_attribute = {}
    for thing, answer in zip(things, answers):
        if isinstance(answer, Exception):
            print(f"!! naming the properties of {thing} failed: {answer}")
            continue
        names = {}
        apply_names(answer, "a", list(values_of[thing]), names)
        for attribute, canonical in names.items():
            canonical_attribute[(thing, attribute)] = snake_case(canonical)

    updates = []
    for statement_id, entity, tag, attribute, _ in rows:
        thing = canonical_entity[(entity, tag or "")]
        updates.append((thing, canonical_attribute.get((thing, attribute), snake_case(attribute)),
                        statement_id))
    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE statements SET canonical_entity = %s, canonical_attribute = %s WHERE statement_id = %s",
            updates,
        )
    for (entity, tag), thing in canonical_entity.items():
        print(f"   entity    {entity!r} [{tag}] -> {thing!r}")
    for (thing, attribute), canonical in canonical_attribute.items():
        if canonical != attribute:
            print(f"   attribute {thing}: {attribute!r} -> {canonical!r}")
    return len(set(canonical_entity.values())), len({(t, a) for t, a, _ in updates})


# --------------------------------------------------------------------------
# Working out what an addendum has replaced
# --------------------------------------------------------------------------

class Amending(BaseModel):
    """Whether a piece of text issues a binding change to the pack."""

    is_amendment: bool = Field(
        description="True only if this text itself changes what the pack requires."
    )
    reason: str = Field(description="One short sentence saying why.")


AMENDMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Tender packs are corrected over time by addenda. Decide whether the text below "
     "is itself a binding change to the pack.\n"
     "Answer true when the text issues a change, for example by numbering itself as an "
     "addendum item and saying that a value is replaced, superseded, deleted or now reads "
     "differently.\n"
     "Answer false for everything else, including:\n"
     "- the original requirements themselves, which state rules but change nothing;\n"
     "- answers to questions, clarifications and other text that is marked as information "
     "only or that is not adopted by a numbered addendum;\n"
     "- lists of what an addendum did NOT change;\n"
     "- covers, contents tables, acknowledgements and records of meetings or visits.\n"
     "A document can hold both binding and non-binding parts, so judge this text alone, "
     "not the document it sits in."),
    ("human", "Document: {doc}\nPage: {page}\nSection: {section}\n\n{content}"),
])


class Replaced(BaseModel):
    replaced_statement_ids: list[str] = Field(
        description="Ids of the statements this amendment replaces. Empty if it replaces none."
    )
    reason: str = Field(description="One short sentence saying what was replaced and why.")


REPLACED_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "An addendum has changed the pack. You are given the amendment and a list of existing "
     "statements. Say which of those statements the amendment has replaced.\n"
     "A statement is replaced only when the amendment sets a new value for the SAME property "
     "of the SAME thing, so that the old statement must no longer be followed.\n"
     "Rules:\n"
     "- Do not replace a statement about a different property, a different room or a "
     "different piece of equipment.\n"
     "- Do not replace a statement just because the amendment mentions it, refers to its "
     "section, or says it is unchanged.\n"
     "- One amendment can replace several statements in several documents; list them all.\n"
     "- An earlier addendum's statement can itself be replaced by a later one.\n"
     "- When the amendment replaces nothing, return an empty list.\n"
     "- Only use ids from the list you were given."),
    ("human",
     "Amendment ({doc} p{page}, {section}):\n"
     "{entity} | {attribute} = {value}\n\"{quote}\"\n\n"
     "Full text of the addendum item:\n{content}\n\n"
     "Existing statements:\n{candidates}"),
])


def amendment_chain():
    llm = ChatOpenAI(model=CHAT_MODEL, base_url=OPENAI_API_BASE,
                     api_key=OPENAI_API_KEY, temperature=0, max_retries=3)
    return AMENDMENT_PROMPT | llm.with_structured_output(Amending, method="json_schema")


def replaced_chain():
    llm = ChatOpenAI(model=CHAT_MODEL, base_url=OPENAI_API_BASE,
                     api_key=OPENAI_API_KEY, temperature=0, max_retries=3)
    return REPLACED_PROMPT | llm.with_structured_output(Replaced, method="json_schema")


def find_amendment_chunks(chunks):
    """Which chunks actually change the pack. Judged per chunk, not per document."""
    results = amendment_chain().batch(
        [
            {
                "doc": chunk["doc"],
                "page": chunk["page"],
                "section": " > ".join(
                    h for h in (chunk["page_title"], chunk["section_parent"], chunk["section_heading"]) if h
                ),
                "content": chunk["section_content"],
            }
            for chunk in chunks
        ],
        config={"max_concurrency": EXTRACT_CONCURRENCY},
        return_exceptions=True,
    )
    amending = set()
    for chunk, result in zip(chunks, results):
        if isinstance(result, Exception):
            print(f"!! {chunk['chunk_id']} amendment check failed: {result}")
            continue
        if result.is_amendment:
            amending.add(chunk["chunk_id"])
            print(f"++ amendment {chunk['chunk_id']} {chunk['doc']} p{chunk['page']}: {result.reason}")
    return amending


# --------------------------------------------------------------------------
# Finding contradictions
# --------------------------------------------------------------------------

class Verdict(BaseModel):
    """What the judge decides about one group of statements."""

    explanation: str = Field(description="One or two sentences comparing the values and saying what follows.")
    report: bool = Field(
        description="True when the statements that still apply disagree, or when a document "
                    "still states a value an addendum replaced."
    )
    statement_ids: list[str] = Field(
        description="Ids of every statement that takes part, from the list given."
    )
    category: Literal["numeric", "unit", "scope", "date_arithmetic", "omission",
                      "terminology", "superseded_text", "other"]
    severity: Literal["high", "medium", "low"]
    confidence: float = Field(description="How sure you are this is a real contradiction, 0 to 1.")


JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You check a tender pack for contradictions. Below is everything the documents say "
     "about one property of one thing, one statement per line, marked with its status and "
     "kind. Decide whether to report them.\n"
     "Report them when a contractor could not follow all the statements that still apply, "
     "and also when a document still states a value that an addendum has replaced: the "
     "reader must know that this text no longer counts.\n"
     "How to judge:\n"
     "- Convert units and do the arithmetic first. Values that are equal once converted "
     "agree, however differently they are written.\n"
     "- The statements come from different documents on purpose: a disagreement between "
     "documents is exactly what we are looking for.\n"
     "- These are not contradictions: a preference or a permission next to a requirement; "
     "a provided quantity above a minimum; different wording with the same meaning; one "
     "statement giving more detail than another without disagreeing.\n"
     "- 'superseded' statements were replaced by an addendum. An earlier addendum replaced "
     "by a later one is how addenda work, and is not a contradiction on its own.\n"
     "- 'information' statements, such as answers to tender queries, are not binding. They "
     "never decide a contradiction, but list them when they confirm one side.\n"
     "- 'worked_out' statements were calculated from a table, e.g. the day a programme "
     "finishes is its latest activity finish. They count like the rows they come from.\n"
     "If you report them, list every statement that takes part and choose the category:\n"
     "- superseded_text: the statements that still apply agree, but a document still states "
     "a value an addendum replaced;\n"
     "- date_arithmetic: dates or durations that do not fit together, e.g. a programme that "
     "ends after the completion date;\n"
     "- numeric: numbers that differ; unit: units that differ or are missing;\n"
     "- scope: who supplies or does what; terminology: names or definitions that differ;\n"
     "- omission: a list that claims to be complete leaves out something another statement "
     "requires; other: anything else.\n"
     "Severity is about consequence: cost, safety or programme impact is high; wording and "
     "differences an addendum already settled are low."),
    ("human", "Thing: {thing}\nProperty: {property}\n\n{statements}"),
])


def judge_chain():
    llm = ChatOpenAI(model=CHAT_MODEL, base_url=OPENAI_API_BASE,
                     api_key=OPENAI_API_KEY, temperature=0, max_retries=3)
    return JUDGE_PROMPT | llm.with_structured_output(Verdict, method="json_schema")


def section_of(statement):
    return " > ".join(h for h in (statement["section_parent"], statement["section_heading"]) if h)


def describe_for_judge(statement):
    return (f"{statement['statement_id']} [{statement['status']}, {statement['kind']}] "
            f"{statement['normal_value'] or '(none)'} | {statement['doc']} p{statement['page']}, "
            f"{section_of(statement)} | \"{statement['quote']}\"")


def disagreeing_groups(statements):
    """Statements grouped by thing and property, keeping only the groups whose values
    differ. A group that agrees everywhere can not hold a contradiction."""
    groups = {}
    for statement in statements:
        groups.setdefault((statement["entity"], statement["attribute"]), []).append(statement)
    return [group for group in groups.values()
            if len({statement["normal_value"] for statement in group}) > 1]


def describe(statement):
    """One line per statement, as the model sees it."""
    return (f"{statement['statement_id']} | {statement['entity']} | "
            f"{statement['attribute']} = {statement['normal_value'] or '(none)'} | "
            f"{statement['doc']} p{statement['page']}, {section_of(statement)} | "
            f"\"{statement['quote']}\"")


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------

def ensure_schema(conn):
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS chunk_seq")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY
                DEFAULT 'chunk_' || lpad(nextval('chunk_seq')::text, 6, '0'),
            doc TEXT NOT NULL,
            page INTEGER NOT NULL,
            page_title TEXT NOT NULL DEFAULT '',
            section_parent TEXT NOT NULL DEFAULT '',
            section_heading TEXT NOT NULL DEFAULT '',
            section_content TEXT NOT NULL,
            embedding vector(%s)
        )
    """ % EMBEDDING_DIM)
    conn.execute("CREATE SEQUENCE IF NOT EXISTS statement_seq")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS statements (
            statement_id TEXT PRIMARY KEY
                DEFAULT 'statement_' || lpad(nextval('statement_seq')::text, 6, '0'),
            entity TEXT NOT NULL,
            canonical_entity TEXT,
            equipment_tag TEXT,
            attribute TEXT NOT NULL,
            value TEXT,
            unit TEXT,
            quote TEXT NOT NULL,
            source_chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
            status TEXT NOT NULL DEFAULT 'active',
            superseded_by_statement_id TEXT
        )
    """)
    conn.execute("ALTER TABLE statements ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'requirement'")
    conn.execute("ALTER TABLE statements ADD COLUMN IF NOT EXISTS normal_value TEXT")
    conn.execute("ALTER TABLE statements ADD COLUMN IF NOT EXISTS canonical_attribute TEXT")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS finding_seq")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS findings (
            finding_id TEXT PRIMARY KEY
                DEFAULT 'finding_' || lpad(nextval('finding_seq')::text, 6, '0'),
            entity TEXT NOT NULL,
            attribute TEXT NOT NULL,
            statement_ids TEXT[] NOT NULL,
            category TEXT NOT NULL,
            severity TEXT NOT NULL,
            confidence REAL NOT NULL,
            explanation TEXT NOT NULL,
            review_status TEXT NOT NULL DEFAULT 'unreviewed',
            review_note TEXT
        )
    """)


def save_chunks(conn, filename, chunks, vectors):
    """Replace this document's chunks with the ones we just extracted."""
    conn.execute("DELETE FROM chunks WHERE doc = %s", (filename,))
    saved = []
    with conn.cursor() as cur:
        for chunk, vector in zip(chunks, vectors):
            cur.execute(
                """INSERT INTO chunks (doc, page, page_title, section_parent, section_heading,
                                       section_content, embedding)
                   VALUES (%(doc)s, %(page)s, %(page_title)s, %(section_parent)s,
                           %(section_heading)s, %(section_content)s, %(embedding)s::vector)
                   RETURNING chunk_id""",
                chunk | {"embedding": to_vector(vector)},
            )
            saved.append(cur.fetchone()[0])
    return saved


# --------------------------------------------------------------------------
# Web
# --------------------------------------------------------------------------

@app.get("/")
def index(request: Request):
    """Show the upload form."""
    return templates.TemplateResponse(request, "index.html")


@app.post("/uploads", response_class=PlainTextResponse)
async def uploads(files: list[UploadFile] = File(...)):
    """Chunk every uploaded PDF and store the chunks in PostgreSQL."""
    lines = []
    with psycopg.connect(DATABASE_URL) as conn:
        ensure_schema(conn)
        for file in files:
            content = await file.read()
            chunks = make_chunks(content, file.filename)
            vectors = embed([embedding_text(chunk) for chunk in chunks])
            ids = save_chunks(conn, file.filename, chunks, vectors)
            print(f"--- {file.filename}: {len(chunks)} chunks ---")
            for chunk_id, chunk in zip(ids, chunks):
                print(f"[{chunk_id}] p{chunk['page']} "
                      f"{chunk['section_parent']} > {chunk['section_heading']}")
                print(f"    {chunk['section_content'][:120]}...")
            lines.append(f"{file.filename}: {len(chunks)} chunks embedded and stored")
    return "\n".join(lines)


@app.post("/extract", response_class=PlainTextResponse)
def extract():
    """Read every stored chunk and record the statements it makes."""
    with psycopg.connect(DATABASE_URL) as conn:
        ensure_schema(conn)
        chunks = conn.execute(
            """SELECT chunk_id, doc, page, page_title, section_parent, section_heading, section_content
               FROM chunks ORDER BY chunk_id"""
        ).fetchall()
        columns = ["chunk_id", "doc", "page", "page_title", "section_parent",
                   "section_heading", "section_content"]
        chunks = [dict(zip(columns, row)) for row in chunks]
        if not chunks:
            return "No chunks stored yet. Upload the pack first."

        found = extract_statements(chunks)

        conn.execute("DELETE FROM findings")  # they point at the statements about to go
        conn.execute("DELETE FROM statements")
        unverified, skipped = 0, 0
        with conn.cursor() as cur:
            for chunk, statement in found:
                if statement.kind == "document_control":
                    # Revisions, issue dates and the like are about the paperwork, not the
                    # Works, so they are never compared.
                    skipped += 1
                    print(f"   skip {statement.entity} | {statement.attribute} = {statement.value} "
                          f"({chunk['doc']} p{chunk['page']})")
                    continue
                if not quote_matches(statement.quote, chunk["section_content"]):
                    unverified += 1
                    print(f"?? quote not found in {chunk['chunk_id']}: {statement.quote!r}")
                normal_value = normalise(statement.value, statement.unit)
                cur.execute(
                    """INSERT INTO statements (entity, equipment_tag, attribute, value, unit,
                                               normal_value, quote, kind, source_chunk_id)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                       RETURNING statement_id""",
                    (statement.entity, statement.equipment_tag, statement.attribute,
                     statement.value, statement.unit, normal_value, statement.quote,
                     statement.kind, chunk["chunk_id"]),
                )
                print(f"[{cur.fetchone()[0]}] {statement.kind}: {statement.entity} | "
                      f"{statement.attribute} = {normal_value} ({chunk['doc']} p{chunk['page']})")

            # Stored before naming, so they join the contract's own completion date and duration.
            facts = programme_facts(chunks)
            for chunk, attribute, value, unit, quote in facts:
                normal_value = normalise(value, unit)
                cur.execute(
                    """INSERT INTO statements (entity, attribute, value, unit, normal_value, quote,
                                               kind, source_chunk_id)
                       VALUES ('Works', %s, %s, %s, %s, %s, 'worked_out', %s)""",
                    (attribute, value, unit, normal_value, quote, chunk["chunk_id"]),
                )
                print(f"[worked out] Works | {attribute} = {normal_value} ({chunk['doc']} p{chunk['page']})")

        thing_count, pair_count = name_consistently(conn)

    return (f"{len(found)} statements extracted from {len(chunks)} chunks"
            f"\n{skipped} about the documents themselves, not stored"
            f"\n{unverified} quote(s) could not be found in their source chunk"
            f"\n{len(facts)} fact(s) worked out from programmes"
            f"\n{thing_count} distinct things and {pair_count} distinct thing + property pairs")


# Every later step sees the canonical names; the names as written stay in the table.
STATEMENT_SQL = """
    SELECT s.statement_id, COALESCE(s.canonical_entity, s.entity),
           COALESCE(s.canonical_attribute, s.attribute), s.value, s.unit, s.normal_value, s.quote,
           s.kind, s.source_chunk_id, s.status, c.doc, c.page, c.page_title,
           c.section_parent, c.section_heading, c.section_content
    FROM statements s JOIN chunks c ON c.chunk_id = s.source_chunk_id
"""
STATEMENT_COLUMNS = ["statement_id", "entity", "attribute", "value", "unit", "normal_value",
                     "quote", "kind", "chunk_id", "status", "doc", "page", "page_title",
                     "section_parent", "section_heading", "section_content"]


def fetch_statements(conn, where="", args=()):
    rows = conn.execute(STATEMENT_SQL + where, args).fetchall()
    return [dict(zip(STATEMENT_COLUMNS, row)) for row in rows]


@app.post("/supersede", response_class=PlainTextResponse)
def supersede():
    """Find the addenda, and mark the statements they have replaced."""
    with psycopg.connect(DATABASE_URL) as conn:
        ensure_schema(conn)
        statements = fetch_statements(conn, " ORDER BY c.doc, c.page, s.statement_id")
        if not statements:
            return "No statements yet. Run Extract first."

        # Only the chunks that carry statements are worth judging.
        seen, chunks = set(), []
        for statement in statements:
            if statement["chunk_id"] not in seen:
                seen.add(statement["chunk_id"])
                chunks.append(statement)
        amending_chunks = find_amendment_chunks(chunks)
        amendments = [s for s in statements if s["chunk_id"] in amending_chunks]

        # Re-running must start from a clean slate.
        conn.execute("UPDATE statements SET status = 'active', superseded_by_statement_id = NULL")

        # An amendment can only replace what already existed when it was issued, so each
        # one is given a position and may not touch itself or any later amendment.
        position = {}
        for index, chunk_id in enumerate(dict.fromkeys(a["chunk_id"] for a in amendments)):
            position[chunk_id] = index

        replaced_count = 0
        chain = replaced_chain()
        # Oldest amendment first, so a later addendum overwrites an earlier one's decision.
        for amendment in amendments:
            mine = position[amendment["chunk_id"]]
            later = [chunk_id for chunk_id, index in position.items() if index >= mine]
            # Candidates: everything said about the same thing, then the statements of the
            # chunks closest in meaning, in case the thing was named differently.
            candidates = fetch_statements(
                conn,
                """ WHERE s.source_chunk_id <> ALL(%s)
                      AND COALESCE(s.canonical_entity, s.entity) = %s""",
                (later, amendment["entity"]),
            )
            vector = to_vector(embed([describe(amendment)])[0])
            nearest = fetch_statements(
                conn,
                """ WHERE s.source_chunk_id <> ALL(%s)
                    ORDER BY (SELECT embedding FROM chunks WHERE chunk_id = s.source_chunk_id)
                             <=> %s::vector
                    LIMIT %s""",
                (later, vector, CANDIDATE_LIMIT),
            )
            same_thing = {c["statement_id"] for c in candidates}
            candidates += [c for c in nearest if c["statement_id"] not in same_thing]
            if not candidates:
                continue
            by_id = {c["statement_id"]: c for c in candidates}

            answer = chain.invoke({
                "doc": amendment["doc"],
                "page": amendment["page"],
                "section": section_of(amendment),
                "content": amendment["section_content"],
                "entity": amendment["entity"],
                "attribute": amendment["attribute"],
                "value": amendment["normal_value"] or "(none)",
                "quote": amendment["quote"],
                "candidates": "\n".join(describe(c) for c in candidates),
            })
            replaced = [by_id[i] for i in answer.replaced_statement_ids if i in by_id]
            if not replaced:
                continue

            for old in replaced:
                conn.execute(
                    """UPDATE statements SET status = 'superseded',
                                             superseded_by_statement_id = %s
                       WHERE statement_id = %s""",
                    (amendment["statement_id"], old["statement_id"]),
                )
                replaced_count += 1
                print(f"-- {old['statement_id']} ({old['entity']} {old['attribute']}="
                      f"{old['value']}) superseded by {amendment['statement_id']}")

            # An amendment sets the same property of the same thing as what it replaces, so
            # they all take one name: the one most of the replaced statements already use.
            names = [(old["entity"], old["attribute"]) for old in replaced]
            entity, attribute = max(set(names), key=names.count)
            conn.execute(
                """UPDATE statements SET canonical_entity = %s, canonical_attribute = %s
                   WHERE statement_id = ANY(%s)""",
                (entity, attribute, [amendment["statement_id"]] + [old["statement_id"] for old in replaced]),
            )
            print(f"   now all named {entity} . {attribute}")

        active = conn.execute(
            "SELECT count(*) FROM statements WHERE status = 'active'"
        ).fetchone()[0]

    return (f"{len(amendments)} amending statement(s) found in {len(amending_chunks)} chunk(s)"
            f"\n{replaced_count} statement(s) marked superseded"
            f"\n{active} statement(s) still active")


SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


@app.post("/scan")
def scan():
    """Compare what the documents say about each thing and record the contradictions."""
    with psycopg.connect(DATABASE_URL) as conn:
        ensure_schema(conn)
        statements = fetch_statements(conn, " ORDER BY s.statement_id")
        if not statements:
            return JSONResponse({"findings": [], "message": "No statements yet. Run Extract first."})

        by_id = {s["statement_id"]: s for s in statements}
        groups = disagreeing_groups(statements)
        print(f"{len(groups)} group(s) of statements give different values")

        verdicts = judge_chain().batch(
            [{"thing": group[0]["entity"], "property": group[0]["attribute"],
              "statements": "\n".join(describe_for_judge(s) for s in group)}
             for group in groups],
            config={"max_concurrency": EXTRACT_CONCURRENCY},
            return_exceptions=True,
        )

        conn.execute("DELETE FROM findings")
        with conn.cursor() as cur:
            for group, verdict in zip(groups, verdicts):
                if isinstance(verdict, Exception):
                    print(f"!! judging {group[0]['entity']} . {group[0]['attribute']} failed: {verdict}")
                    continue
                print(f"{'XX' if verdict.report else 'ok'} {group[0]['entity']} . "
                      f"{group[0]['attribute']}: {verdict.explanation}")
                if not verdict.report:
                    continue
                # Only ids from this group count; if the model named fewer than two, the
                # whole group is shown, since its values differ by construction.
                ids = [s["statement_id"] for s in group if s["statement_id"] in verdict.statement_ids]
                if len(ids) < 2:
                    ids = [s["statement_id"] for s in group]
                cur.execute(
                    """INSERT INTO findings (entity, attribute, statement_ids, category, severity,
                                             confidence, explanation)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (group[0]["entity"], group[0]["attribute"], ids, verdict.category,
                     verdict.severity, min(max(verdict.confidence, 0.0), 1.0), verdict.explanation),
                )

        rows = conn.execute("""
            SELECT finding_id, entity, attribute, statement_ids, category, severity, confidence,
                   explanation, review_status, review_note
            FROM findings
        """).fetchall()

    columns = ["finding_id", "entity", "attribute", "statement_ids", "category", "severity",
               "confidence", "explanation", "review_status", "review_note"]
    findings = []
    for row in rows:
        finding = dict(zip(columns, row))
        # Each conflicting statement, quoted, with where it is in the pack.
        finding["statements"] = [
            {"statement_id": by_id[i]["statement_id"], "doc": by_id[i]["doc"],
             "page": by_id[i]["page"], "section": section_of(by_id[i]),
             "quote": by_id[i]["quote"], "value": by_id[i]["normal_value"],
             "status": by_id[i]["status"], "kind": by_id[i]["kind"]}
            for i in finding.pop("statement_ids")
        ]
        findings.append(finding)
    # Severity first, then confidence: a certain cosmetic issue stays below an uncertain costly one.
    findings.sort(key=lambda f: (SEVERITY_ORDER.get(f["severity"], 3), -f["confidence"]))
    print(f"{len(findings)} finding(s) stored")
    return JSONResponse({"findings": findings})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)
