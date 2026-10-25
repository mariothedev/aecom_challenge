"""Step 1: upload tender PDFs and log what arrived on the server."""

from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import PlainTextResponse
from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Tender contradiction detection")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/")
def index(request: Request):
    """Show the upload form."""
    return templates.TemplateResponse(request, "index.html")


@app.post("/uploads", response_class=PlainTextResponse)
async def uploads(files: list[UploadFile] = File(...)):
    """Read each uploaded file and print what we received."""
    lines = []
    for file in files:
        content = await file.read()
        preview = content[:60]
        print(f"--- {file.filename} ---")
        print(f"content_type: {file.content_type}")
        print(f"size: {len(content)} bytes")
        print(f"first bytes: {preview!r}")
        lines.append(f"{file.filename}  {file.content_type}  {len(content)} bytes")
    return "Received:\n" + "\n".join(lines)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7777)
