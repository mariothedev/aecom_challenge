"""Time embedding models on the same texts.

    python benchmarks/benchmark_embedding.py

Qwen3-Embedding-0.6B on DeepInfra is what main.py uses. The others are from DeepInfra and Azure.
The database stores 1024 numbers per vector; the dims column shows which models fit it as it is.
Needs DEEPINFRA_API_KEY, AZURE_API_KEY and AZURE_ENDPOINT in .env.
"""

import asyncio
import os
from pathlib import Path
from statistics import median
from time import perf_counter

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

DEEPINFRA_URL = "https://api.deepinfra.com/v1/openai"
DEEPINFRA_MODELS = ["Qwen/Qwen3-Embedding-0.6B", "BAAI/bge-m3", "google/embeddinggemma-2",
                    "google/embeddinggemma-300m", "nvidia/Nemotron-3-Embed-1B-NVFP4"]
MARKDOWN_FILES = ["01_Club_Rules_and_Requirements.md", "02_Rooms_and_Equipment_Specification.md",
                  "03_Addenda_and_Query_Responses.md"]
BATCH_SIZE = 32  # about one page of statements
RUNS = 5

# Each one takes a list of texts and gives back one vector per text.
# check_embedding_ctx_length=False sends plain text; otherwise LangChain sends OpenAI token ids.
embedders = {}
for model in DEEPINFRA_MODELS:
    embedders[f"deepinfra {model}"] = OpenAIEmbeddings(
        model=model, base_url=DEEPINFRA_URL, api_key=os.environ["DEEPINFRA_API_KEY"],
        check_embedding_ctx_length=False).aembed_documents
# Azure wants the deployment name; text-embedding-3-small gives 1536 numbers per vector unless asked for fewer
embedders["azure text-embedding-3-small"] = OpenAIEmbeddings(
    model="text-embedding-3-small", base_url=os.environ["AZURE_ENDPOINT"], api_key=os.environ["AZURE_API_KEY"],
    dimensions=1024, check_embedding_ctx_length=False).aembed_documents

# Every line of the club PDFs' markdown, standing in for the statements main.py embeds
texts = []
for name in MARKDOWN_FILES:
    lines = (BASE_DIR / "markdown" / name).read_text().splitlines()
    texts += [line for line in lines if len(line) > 20]  # skip blank lines, page markers and table borders
batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
NAME_WIDTH = max(len(name) for name in embedders) + 2


async def time_it(embed, batches):
    """Seconds to embed every batch, all at once (like main.py: one request per page, all pages at once)."""
    start = perf_counter()
    await asyncio.gather(*[embed(batch) for batch in batches])
    return perf_counter() - start


async def benchmark(name, embed):
    # The first call may also wake the server up
    start = perf_counter()
    vector = (await embed([texts[0]]))[0]
    first = perf_counter() - start

    one_text = median([await time_it(embed, [[texts[0]]]) for _ in range(RUNS)])
    all_texts = median([await time_it(embed, batches) for _ in range(RUNS)])
    print(f"{name:<{NAME_WIDTH}}{len(vector):>6}{first:>8.3f}{one_text:>8.3f}{all_texts:>11.3f}")


async def main():
    print(f"Seconds. first = one run, the rest = median of {RUNS}. "
          f"{len(texts)} texts = {len(batches)} requests sent at once.\n")
    print(f"{'model':<{NAME_WIDTH}}{'dims':>6}{'first':>8}{'1 text':>8}{f'{len(texts)} texts':>11}")
    for name, embed in embedders.items():
        await benchmark(name, embed)


asyncio.run(main())
