"""Gemini embedding factory."""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import get_settings


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    settings = get_settings()
    if not settings.gemini_api_key or settings.gemini_api_key.startswith("your_"):
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Copy .env.example to .env and add your key."
        )
    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=settings.gemini_api_key,
    )
