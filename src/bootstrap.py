"""Ensure the Chroma collection is populated before serving queries."""

from __future__ import annotations

from src.vectorstore import collection_count


def ensure_knowledge_base() -> int:
    """Ingest markdown docs if the persistent collection is empty."""
    count = collection_count()
    if count > 0:
        return count
    from ingest import ingest

    return ingest()
