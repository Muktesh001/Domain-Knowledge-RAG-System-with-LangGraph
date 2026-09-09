"""FastAPI wrapper around the same LangGraph RAG pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.bootstrap import ensure_knowledge_base
from src.config import get_settings
from src.graph.workflow import run_rag

@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    if not settings.gemini_api_key or settings.gemini_api_key.startswith("your_"):
        raise RuntimeError("GEMINI_API_KEY is not set in .env")
    ensure_knowledge_base()
    yield


app = FastAPI(
    title="AI Interview Knowledge RAG API",
    description="LangGraph + Gemini + ChromaDB retrieval-augmented answers.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, examples=["What is LangGraph?"])
    k: int = Field(4, ge=1, le=10)
    temperature: float = Field(0.2, ge=0.0, le=1.0)


class AskResponse(BaseModel):
    question: str
    rewritten_query: str
    answer: str
    documents: list[dict]
    steps: list[str]
    retrieval_ok: bool
    rewrite_attempts: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "ai-interview-rag"}


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest) -> AskResponse:
    try:
        result = run_rag(body.question, k=body.k, temperature=body.temperature)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AskResponse(
        question=result["question"],
        rewritten_query=result["rewritten_query"],
        answer=result["answer"],
        documents=result["documents"],
        steps=result["steps"],
        retrieval_ok=result["retrieval_ok"],
        rewrite_attempts=result["rewrite_attempts"],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
