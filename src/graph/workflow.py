"""Compile and run the LangGraph RAG workflow."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from src.config import get_settings
from src.graph.nodes import add_citations, generate_answer, rerank, retrieve, rewrite_query, route_after_rerank
from src.graph.state import GraphState


def build_rag_graph():
    """Build: rewrite -> retrieve -> rerank -> (rewrite | generate) -> citations."""
    graph = StateGraph(GraphState)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("add_citations", add_citations)

    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_conditional_edges(
        "rerank",
        route_after_rerank,
        {"rewrite": "rewrite_query", "generate": "generate_answer"},
    )
    graph.add_edge("generate_answer", "add_citations")
    graph.add_edge("add_citations", END)
    return graph.compile()


_APP = None


def get_app():
    global _APP
    if _APP is None:
        _APP = build_rag_graph()
    return _APP


def run_rag(question: str, k: int = 4, temperature: float = 0.2) -> dict[str, Any]:
    """Execute the full graph and return a serializable result dict."""
    settings = get_settings()
    initial: GraphState = {
        "question": question,
        "rewritten_query": "",
        "k": k,
        "temperature": temperature,
        "rewrite_attempts": 0,
        "max_rewrite_attempts": settings.max_rewrite_attempts,
        "documents": [],
        "retrieval_ok": False,
        "answer": "",
        "cited_answer": "",
        "steps": [],
        "error": "",
    }
    result = get_app().invoke(initial)
    return {
        "question": result.get("question", question),
        "rewritten_query": result.get("rewritten_query", ""),
        "answer": result.get("cited_answer") or result.get("answer", ""),
        "raw_answer": result.get("answer", ""),
        "documents": result.get("documents", []),
        "steps": result.get("steps", []),
        "retrieval_ok": result.get("retrieval_ok", False),
        "rewrite_attempts": result.get("rewrite_attempts", 0),
    }
