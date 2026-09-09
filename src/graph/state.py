"""Shared LangGraph state for the RAG workflow."""

from __future__ import annotations

from typing import TypedDict


class RetrievedChunk(TypedDict):
    content: str
    source: str
    title: str
    score: float
    relevance: float


class GraphState(TypedDict):
    question: str
    rewritten_query: str
    k: int
    temperature: float
    rewrite_attempts: int
    max_rewrite_attempts: int
    documents: list[RetrievedChunk]
    retrieval_ok: bool
    answer: str
    cited_answer: str
    steps: list[str]
    error: str
