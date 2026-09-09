"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root (parent of src/)
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration for the RAG system."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-3.6-flash"
    gemini_embedding_model: str = "models/gemini-embedding-001"

    chroma_persist_dir: str = str(ROOT_DIR / "chroma_db")
    chroma_collection_name: str = "ai_interview_knowledge"

    knowledge_base_dir: str = str(ROOT_DIR / "data" / "knowledge_base")

    # Retrieval defaults (overridable from Streamlit sidebar)
    default_k: int = 4
    default_temperature: float = 0.2
    max_rewrite_attempts: int = 2
    min_relevance_score: float = 0.35


@lru_cache
def get_settings() -> Settings:
    return Settings()
