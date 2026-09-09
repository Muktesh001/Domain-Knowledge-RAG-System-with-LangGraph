"""Lightweight text cleaning before embeddings are created."""

from __future__ import annotations

import re
import unicodedata


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_HTML_TAG = re.compile(r"<[^>]+>")


def clean_text(text: str) -> str:
    """Normalize unicode, strip noise, and collapse excessive whitespace.

    This is intentionally conservative so interview content and code-like
    terms (e.g. LangGraph, BERT) are preserved.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\ufeff", "").replace("\u200b", "")
    text = _HTML_TAG.sub(" ", text)
    text = _CONTROL_CHARS.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def is_useful_chunk(text: str, min_chars: int = 80) -> bool:
    """Drop tiny or near-empty fragments that would pollute retrieval."""
    stripped = text.strip()
    if len(stripped) < min_chars:
        return False
    alpha = sum(ch.isalpha() for ch in stripped)
    return alpha / max(len(stripped), 1) >= 0.35
