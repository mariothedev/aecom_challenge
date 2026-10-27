"""Time two ways of turning a PDF into markdown with the same LLM.

    python3 llm_speed.py

1. png: render each page to a PNG and send every page at once, like main.py does
2. pdf: send the whole PDF file in one call
"""

import asyncio
import base64
from time import perf_counter

from langchain_core.messages import HumanMessage

from main import BASE_DIR, READ_PAGE_PROMPT, convert_pdf_to_png, get_markdown_from_png, llm

PDF_PATH = BASE_DIR / "data" / "01_Club_Rules_and_Requirements.pdf"
READ_PDF_PROMPT = READ_PAGE_PROMPT.replace("this page", "every page")


async def get_markdown_from_pdf(name, pdf_bytes):
    file_data = "data:application/pdf;base64," + base64.b64encode(pdf_bytes).decode()
    message = HumanMessage(content=[
        {"type": "text", "text": READ_PDF_PROMPT},
        {"type": "file", "file": {"filename": name, "file_data": file_data}},
    ])
    reply = await llm.ainvoke([message])
    return reply.content


async def main():
    pdf_bytes = PDF_PATH.read_bytes()

    start = perf_counter()
    pages = await asyncio.gather(*[get_markdown_from_png(png) for png in convert_pdf_to_png(pdf_bytes)])
    markdown = "\n\n".join(pages)
    print(f"png: {perf_counter() - start:.3f} seconds, {len(pages)} pages, {len(markdown)} characters")

    start = perf_counter()
    markdown = await get_markdown_from_pdf(PDF_PATH.name, pdf_bytes)
    print(f"pdf: {perf_counter() - start:.3f} seconds, {len(markdown)} characters")


asyncio.run(main())
