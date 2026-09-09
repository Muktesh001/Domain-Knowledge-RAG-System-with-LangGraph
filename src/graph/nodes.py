"""LangGraph node implementations for the RAG workflow."""

from __future__ import annotations

import json
import re
from typing import Any  # node return types are partial state dicts

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import get_settings
from src.graph.state import GraphState, RetrievedChunk
from src.prompts import (
    ANSWER_SYSTEM,
    CITATION_SYSTEM,
    QUERY_REWRITER_FEW_SHOT,
    QUERY_REWRITER_SYSTEM,
    RERANKER_SYSTEM,
)
from src.vectorstore import similarity_search_with_score


def _llm(temperature: float) -> ChatGoogleGenerativeAI:
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.gemini_chat_model,
        google_api_key=settings.gemini_api_key,
        temperature=temperature,
    )


def _as_text(content: Any) -> str:
    """Normalize Gemini/LangChain message content to a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    text = getattr(content, "text", None)
    if text:
        return str(text)
    return str(content)


def _append_step(state: GraphState, step: str) -> list[str]:
    steps = list(state.get("steps") or [])
    steps.append(step)
    return steps


def rewrite_query(state: GraphState) -> dict[str, Any]:
    """Improve the user query for retrieval using prompt engineering."""
    llm = _llm(temperature=0.0)
    original = state["question"].strip()
    attempt = int(state.get("rewrite_attempts") or 0) + 1

    prompt = (
        f"{QUERY_REWRITER_FEW_SHOT}\n\n"
        f"User: {original}\n"
        "Rewritten:"
    )
    response = llm.invoke(
        [SystemMessage(content=QUERY_REWRITER_SYSTEM), HumanMessage(content=prompt)]
    )
    rewritten = _as_text(response.content).strip().strip('"')
    if not rewritten:
        rewritten = original

    return {
        "rewritten_query": rewritten,
        "rewrite_attempts": attempt,
        "steps": _append_step(
            state, f"Query Rewriter (attempt {attempt}): {rewritten}"
        ),
    }


def retrieve(state: GraphState) -> dict[str, Any]:
    """Fetch nearest chunks from the persistent Chroma collection."""
    query = state.get("rewritten_query") or state["question"]
    k = int(state.get("k") or get_settings().default_k)

    pairs = similarity_search_with_score(query, k=k)
    documents: list[RetrievedChunk] = []
    for doc, distance in pairs:
        # Chroma L2/cosine distance: convert to a 0-1-ish similarity for display
        similarity = 1.0 / (1.0 + float(distance))
        meta = doc.metadata or {}
        documents.append(
            {
                "content": doc.page_content,
                "source": str(meta.get("source", "unknown")),
                "title": str(meta.get("title", "Untitled")),
                "score": round(similarity, 4),
                "relevance": 0.0,
            }
        )

    return {
        "documents": documents,
        "steps": _append_step(state, f"Retriever: fetched {len(documents)} chunk(s)"),
    }


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"\{.*\}", text, re.DOTALL)
    if fenced:
        text = fenced.group(0)
    return json.loads(text)


def rerank(state: GraphState) -> dict[str, Any]:
    """LLM relevance check + sort. Filters weak chunks when possible."""
    settings = get_settings()
    llm = _llm(temperature=0.0)
    question = state.get("rewritten_query") or state["question"]
    docs = list(state.get("documents") or [])

    graded: list[RetrievedChunk] = []
    for doc in docs:
        snippet = doc["content"][:1200]
        human = (
            f"Question: {question}\n\n"
            f"Document title: {doc['title']}\n"
            f"Document:\n{snippet}"
        )
        try:
            response = llm.invoke(
                [SystemMessage(content=RERANKER_SYSTEM), HumanMessage(content=human)]
            )
            parsed = _parse_json_object(_as_text(response.content))
            relevance = float(parsed.get("score", 0.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            relevance = float(doc.get("score") or 0.0)

        graded.append({**doc, "relevance": round(max(0.0, min(1.0, relevance)), 3)})

    graded.sort(key=lambda d: d["relevance"], reverse=True)
    kept = [d for d in graded if d["relevance"] >= settings.min_relevance_score]
    # Keep at least the top 2 so the answer node is never empty after a rewrite miss
    if len(kept) < 2:
        kept = graded[:2]

    retrieval_ok = bool(kept) and (kept[0]["relevance"] >= settings.min_relevance_score)
    return {
        "documents": kept,
        "retrieval_ok": retrieval_ok,
        "steps": _append_step(
            state,
            f"Re-ranker: kept {len(kept)} chunk(s); top relevance={kept[0]['relevance'] if kept else 0}",
        ),
    }


def route_after_rerank(state: GraphState) -> str:
    """Conditional edge: retry rewrite if retrieval is weak."""
    attempts = int(state.get("rewrite_attempts") or 0)
    max_attempts = int(state.get("max_rewrite_attempts") or get_settings().max_rewrite_attempts)
    if not state.get("retrieval_ok") and attempts < max_attempts:
        return "rewrite"
    return "generate"


def generate_answer(state: GraphState) -> dict[str, Any]:
    """Grounded answer generation with Gemini."""
    temperature = float(state.get("temperature") or get_settings().default_temperature)
    llm = _llm(temperature=temperature)
    docs = state.get("documents") or []

    if not docs:
        return {
            "answer": (
                "I could not find relevant material in the interview knowledge base "
                "for this question. Try rephrasing or ingesting additional documents."
            ),
            "steps": _append_step(state, "Answer Generator: no context"),
        }

    context_blocks = []
    for i, doc in enumerate(docs, start=1):
        context_blocks.append(
            f"[{i}] Title: {doc['title']}\nSource: {doc['source']}\n{doc['content']}"
        )
    context = "\n\n---\n\n".join(context_blocks)
    human = f"Question: {state['question']}\n\nContext:\n{context}"

    response = llm.invoke(
        [SystemMessage(content=ANSWER_SYSTEM), HumanMessage(content=human)]
    )
    answer = _as_text(response.content).strip()
    return {
        "answer": answer,
        "steps": _append_step(state, "Answer Generator: drafted grounded answer"),
    }


def add_citations(state: GraphState) -> dict[str, Any]:
    """Attach inline citations and a sources footer."""
    llm = _llm(temperature=0.0)
    docs = state.get("documents") or []
    answer = state.get("answer") or ""

    source_list = []
    for i, doc in enumerate(docs, start=1):
        snippet = doc["content"][:280].replace("\n", " ")
        source_list.append(
            f"[{i}] title={doc['title']} | file={doc['source']} | snippet={snippet}"
        )

    human = (
        f"Answer:\n{answer}\n\n"
        f"Available sources:\n" + "\n".join(source_list)
    )
    response = llm.invoke(
        [SystemMessage(content=CITATION_SYSTEM), HumanMessage(content=human)]
    )
    cited = _as_text(response.content).strip()
    return {
        "cited_answer": cited,
        "steps": _append_step(state, "Citation Adder: attached sources"),
    }
