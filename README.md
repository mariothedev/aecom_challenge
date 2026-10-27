# Requirements

Create a `.env` file in the project root with these values. They win over any keys already set on your machine:

```dotenv
DATABASE_URL="postgresql://user:password@localhost:5432/dbname"

DEEPINFRA_API_KEY=""
WATSONX_API_KEY=""
LLAMA_CLOUD_API_KEY=""
LLAMA_PARSE=""

AZURE_API_KEY=""
AZURE_ENDPOINT=""
DOCLING_API_KEY=""
```

The database needs the [pgvector](https://github.com/pgvector/pgvector) extension.

# Getting started

After saving `.env`, install the dependencies and start the FastAPI server from the project root:

```bash
pip install -r requirements.txt
python main.py
```

The server runs at <http://localhost:8080>. Click **Upload** and pick one or more PDFs.

# Benchmarks

I used these benchmarks to test various models and their overall performance.

## LLM benchmarks

5 pages sent at once, one run per model.

| Provider  | Model                                | Statements | Seconds |
|-----------|--------------------------------------|-----------:|--------:|
| deepinfra | zai-org/GLM-5.3-Flash                |         70 |   5.496 |
| deepinfra | Qwen/Qwen3-VL-235B-A22B-Instruct     |         69 | 168.869 |
| deepinfra | deepseek-ai/DeepSeek-V4.1-Flash-1017 |         84 |  11.703 |
| deepinfra | google/gemma-4-26B-A4B-it            |         66 |  11.670 |
| azure     | Kimi-K2.6                            |         86 |  70.258 |
| azure     | DeepSeek-V4-Flash                    |         71 |   9.083 |

## Embedding benchmarks

Seconds. First = one run, the rest = median of 5. 188 texts = 6 requests sent at once.

| Provider  | Model                                | Dims | First | 1 text | 188 texts |
|-----------|--------------------------------------|-----:|------:|-------:|----------:|
| deepinfra | Qwen/Qwen3-Embedding-0.6B            | 1024 | 0.862 |  0.305 |     0.697 |
| deepinfra | BAAI/bge-m3                          | 1024 | 0.602 |  0.264 |     0.765 |
| deepinfra | google/embeddinggemma-2              |  768 | 0.637 |  0.339 |     1.336 |
| deepinfra | google/embeddinggemma-300m           |  768 | 0.523 |  0.213 |     0.527 |
| deepinfra | nvidia/Nemotron-3-Embed-1B-NVFP4     | 2048 | 0.644 |  0.314 |     0.807 |
| azure     | text-embedding-3-small               | 1024 | 0.224 |  0.201 |     1.052 |

## OCR benchmarks

Seconds. First = one run, median = median of the next N runs.

| Service        | Pages | Chars |  First | Median |
|----------------|------:|------:|-------:|-------:|
| liteparse      |     5 |  4865 |  0.253 |  0.241 |
| docling        |     5 |  5844 | 24.202 | 24.179 |
| watsonx        |     5 |  5194 | 33.441 | 19.944 |
| llamaparse     |     5 |  6526 |  6.763 |  6.512 |
| mistral-ocr-4-0 |    5 |  5647 |  2.426 |  2.616 |


Note: after careful benchmarking, in terms of extraction quality, I would rate them in this order:

1. VLMs (visual language models) - they gave the best PDF extractions, easy to read and follow.
2. docling - very accurate, but with OCR enabled, you need a GPU in place; otherwise, it runs on CPU and could be somewhat slow.
3. liteparse - gives similar accuracy to docling AND runs on CPU, but benchmarks on the internet show it's less accurate than docling.

best for accuracy: VLMs (Sonnet 5.5, Qwen/Qwen3-VL-235B-A22B-Instruct, etc.)
balanced accuracy with speed: docling, unstructured, liteparse

What I wouldn't use, though:
```
pymupdf - okay, hard to understand
pdfplumber - better than pymupdf, slightly better at tables
```
Though they are super fast, they make extraction harder from an LLM's perspective.


# Summary

For this case, I basically slimmed down the requirements and focused solely on the contradiction engine. Here's basically what I did and why:

## Part one
1. The user uploads their PDFs
2. I give those PDFs to an OCR model
3. The OCR model gives a structured Markdown file per page in the PDF (e.g., 1 PDF with 5 pages = 5 Markdown pages)
4. I give that Markdown file to an LLM and say "find the statements|rules|requirements in this Markdown file"
5. The LLM gives me back a JSON list of "statements"
6. I take those "statements", embed them, store them, and now we have SIMILARITY SEARCH CAPABILITIES!

## Part two
1. Now, here's the fun part. I thought, well, now that we have vectors in our db, why don't we CLUSTER THEM so we know which statements are similar (a good foundation for finding contradictions)? So I looked into the following clustering techniques:
 * DBSCAN/HDBSCAN: slow, some statements are skipped, and it might put related statements in different groups
 * NLI models: bad with numbers, units and dates (bad for our use case, since these are important in the construction industry)
 * K-means: requires knowing the number of groups upfront, not really applicable for our task

 Ultimately, I decided on using nearest neighbours: each statement is paired with its 5 most similar statements. Good enough for this task.

2. Once we have clusters of similar statements, we pass them to the LLM and ask "which ones are contradicting?"
3. We get conflicts!


# TODO

These are things I would consider before going to production:

- Test prompts thoroughly using PromptFoo
- Pros and cons of using Postgres versus a dedicated vector db
- Some sort of pub/sub event system to increase reliability in the system in general (maybe)
- on-demand server-less inference for Docling?
