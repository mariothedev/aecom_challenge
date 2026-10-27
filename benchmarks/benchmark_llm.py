"""Time LLMs on statement extraction: the same pages and the same prompt as main.py.

    python benchmarks/benchmark_llm.py

Every page goes to the model at once, like main.py does, so seconds = the slowest page.
bad pages = pages where the reply was not the JSON we asked for.
Each model's statements are saved to markdown/benchmark_llm/ so you can compare them.
Needs DEEPINFRA_API_KEY, AZURE_API_KEY and AZURE_ENDPOINT in .env.

One PDF and one run, because the Azure deployments allow only 20,000 tokens a minute.
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)
sys.path.append(str(BASE_DIR))  # so main.py, one folder up, can be imported

from main import DEEPINFRA_URL, EXTRACT_PROMPT  # the same prompt main.py uses

MARKDOWN_PATH = BASE_DIR / "markdown" / "01_Club_Rules_and_Requirements.md"  # saved by main.py's ingest
OUTPUT_DIR = BASE_DIR / "markdown" / "benchmark_llm"
DEEPINFRA_MODELS = ["zai-org/GLM-5.3-Flash", "Qwen/Qwen3-VL-235B-A22B-Instruct",
                    "deepseek-ai/DeepSeek-V4.1-Flash-1017", "google/gemma-4-26B-A4B-it"]
AZURE_MODELS = ["Kimi-K2.6", "DeepSeek-V4-Flash"]  # Azure wants the deployment names

# Same settings as main.py: temperature 0 and JSON replies only
llms = {}
for model in DEEPINFRA_MODELS:
    llms[f"deepinfra {model}"] = ChatOpenAI(
        model=model, base_url=DEEPINFRA_URL, api_key=os.environ["DEEPINFRA_API_KEY"],
        temperature=0).bind(response_format={"type": "json_object"})
# Last, so the DeepInfra numbers still print if Azure's limits stop the script
for model in AZURE_MODELS:
    llms[f"azure {model}"] = ChatOpenAI(
        model=model, base_url=os.environ["AZURE_ENDPOINT"], api_key=os.environ["AZURE_API_KEY"],
        temperature=0).bind(response_format={"type": "json_object"})

# main.py saved the PDF's markdown with a "<!-- page N -->" line before each page
pages = re.split(r"<!-- page \d+ -->\n", MARKDOWN_PATH.read_text())[1:]
NAME_WIDTH = max(len(name) for name in llms) + 2


async def extract_statements(llm, page):
    """The statements on one page, or None if the reply is not the JSON we asked for."""
    question = f"{page}\n\nList the statements on this page as JSON."
    reply = await llm.ainvoke([SystemMessage(EXTRACT_PROMPT), HumanMessage(question)])
    try:
        return json.loads(reply.content)["statements"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


async def benchmark(name, llm):
    start = perf_counter()
    statements_per_page = await asyncio.gather(*[extract_statements(llm, page) for page in pages])
    seconds = perf_counter() - start

    file_name = name.replace(" ", "_").replace("/", "_") + ".json"
    (OUTPUT_DIR / file_name).write_text(json.dumps(statements_per_page, indent=2))
    bad_pages = statements_per_page.count(None)
    statements = sum(len(page) for page in statements_per_page if page is not None)
    print(f"{name:<{NAME_WIDTH}}{statements:>12}{bad_pages:>11}{seconds:>9.3f}")


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{MARKDOWN_PATH.name}: {len(pages)} pages sent at once, one run per model.\n")
    print(f"{'model':<{NAME_WIDTH}}{'statements':>12}{'bad pages':>11}{'seconds':>9}")
    for name, llm in llms.items():
        await benchmark(name, llm)


asyncio.run(main())
