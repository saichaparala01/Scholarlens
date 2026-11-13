"""ScholarLens backend: a small FastAPI server around the NLP pipeline.

Run from the project root:
    uvicorn backend.main:app --reload --port 8000
Then open http://localhost:8000 (the frontend is served by this same app) or
http://localhost:8000/docs for the interactive API documentation.
"""

import io
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.pipeline import ScholarLens

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"

app = FastAPI(title="ScholarLens API", version="1.0",
              description="Summarization, entity extraction and question answering for research papers.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

lens = ScholarLens()  # models load lazily on the first request that needs them


class TextIn(BaseModel):
    text: str = Field(..., min_length=1, description="Full text of the paper")


class QuestionIn(TextIn):
    question: str = Field(..., min_length=3)


@app.get("/api/health")
def health():
    return {"status": "ok", "models": lens.status()}


@app.post("/api/summarize")
def summarize(body: TextIn):
    return lens.summarize(body.text)


@app.post("/api/entities")
def entities(body: TextIn):
    return {"entities": lens.entities(body.text)}


@app.post("/api/ask")
def ask(body: QuestionIn):
    return lens.ask(body.question, body.text)


@app.post("/api/extract")
async def extract(file: UploadFile = File(...)):
    """Turn an uploaded PDF or .txt file into plain text for the other endpoints."""
    data = await file.read()
    name = (file.filename or "").lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    elif name.endswith(".txt"):
        text = data.decode("utf-8", errors="ignore")
    else:
        raise HTTPException(status_code=400, detail="Please upload a .pdf or .txt file.")
    if not text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted (scanned PDF?).")
    return {"text": text, "characters": len(text)}


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
