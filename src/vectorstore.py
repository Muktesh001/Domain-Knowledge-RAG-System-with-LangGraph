"""Persistent ChromaDB vector store helpers."""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import get_settings
from src.embeddings import get_embeddings


def get_vectorstore() -> Chroma:
    """Open (or create) the persistent Chroma collection."""
    settings = get_settings()
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.chroma_collection_name,
        persist_directory=settings.chroma_persist_dir,
        embedding_function=get_embeddings(),
    )


def collection_count() -> int:
    try:
        store = get_vectorstore()
        return int(store._collection.count())  # noqa: SLF001
    except Exception:
        return 0


def similarity_search_with_score(query: str, k: int = 4) -> list[tuple[Document, float]]:
    """Return documents with distance scores (lower distance = closer)."""
    store = get_vectorstore()
    return store.similarity_search_with_score(query, k=k)
