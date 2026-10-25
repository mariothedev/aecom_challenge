# Requirements

Create a `.env` file in the project root with these values, or set them on the host machine:

```dotenv
DATABASE_URL="postgresql://user:password@localhost:5432/dbname"
OPENAI_API_KEY="your-openai-api-key"
```

# Getting started

After saving `.env`, install the dependencies and start the FastAPI server from the project root:

```bash
pip install -r requirements.txt
python main.py
```

The server runs at <http://localhost:8080>.
