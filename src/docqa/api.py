"""FastAPI app: POST /ask, GET /health, and a static chat page at /."""
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, rag, store

app = FastAPI(title="SA Document Q&A")
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=config.TOP_K, ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": store.count()}


@app.post("/ask")
def ask(req: AskRequest):
    try:
        return rag.answer(req.question, req.top_k)
    except Exception as exc:  # surface LLM/API errors as a clean 502
        raise HTTPException(status_code=502, detail=f"Answer generation failed: {exc}")


@app.get("/")
def index():
    return FileResponse(config.STATIC_DIR / "index.html")
