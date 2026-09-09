"""Load, clean, chunk, embed, and persist the interview knowledge base."""

from __future__ import annotations

import sys
from pathlib import Path

import chromadb
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Allow `python ingest.py` from the project root
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cleaning import clean_text, is_useful_chunk
from src.config import get_settings
from src.vectorstore import get_vectorstore


def _reset_collection() -> None:
    """Drop the existing Chroma collection so re-ingest is idempotent."""
    settings = get_settings()
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    try:
        client.delete_collection(settings.chroma_collection_name)
    except Exception:
        pass


def load_knowledge_files(kb_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(kb_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        cleaned = clean_text(raw)
        title = path.stem.replace("_", " ").title()
        # First markdown heading becomes a nicer title when present
        for line in cleaned.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        docs.append(
            Document(
                page_content=cleaned,
                metadata={"source": path.name, "title": title, "path": str(path)},
            )
        )
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=140,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(docs)
    return [c for c in chunks if is_useful_chunk(c.page_content)]


def ingest() -> int:
    settings = get_settings()
    kb_dir = Path(settings.knowledge_base_dir)
    if not kb_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

    source_docs = load_knowledge_files(kb_dir)
    if not source_docs:
        raise RuntimeError(f"No markdown files found in {kb_dir}")

    chunks = chunk_documents(source_docs)
    _reset_collection()
    store = get_vectorstore()

    ids = [f"chunk-{i:04d}" for i in range(len(chunks))]
    store.add_documents(chunks, ids=ids)
    print(f"Ingested {len(source_docs)} files -> {len(chunks)} cleaned chunks.")
    print(f"Chroma persist dir: {settings.chroma_persist_dir}")
    return len(chunks)


if __name__ == "__main__":
    ingest()
