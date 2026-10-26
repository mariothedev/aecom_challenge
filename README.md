# Requirements

Create a `.env` file in the project root with these values. They win over any keys already set on your machine:

```dotenv
DATABASE_URL="postgresql://user:password@localhost:5432/dbname"
DEEPINFRA_API_KEY="your-deepinfra-api-key"
```

The database needs the [pgvector](https://github.com/pgvector/pgvector) extension.

# Getting started

After saving `.env`, install the dependencies and start the FastAPI server from the project root:

```bash
pip install -r requirements.txt
python main.py
```

The server runs at <http://localhost:8080>. Click **Upload** and pick one or more PDFs.

# What happens on upload

1. Every page of every PDF is turned into an image and read by `Qwen/Qwen3-VL-235B-A22B-Instruct`, all at the same time. The markdown of each PDF is saved in `markdown/`.
2. Each page of markdown goes back to the LLM, which returns a JSON list of statements (requirements, rules, values, dates, amendments, ...). All pages run at the same time.
3. Each statement is embedded with `Qwen/Qwen3-Embedding-0.6B` and stored in the `statements` table.

Uploading the same file again replaces its statements.
