"""Streamlit chat UI for the AI Interview Knowledge RAG System."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import get_settings
from src.bootstrap import ensure_knowledge_base
from src.graph.workflow import run_rag

st.set_page_config(
    page_title="AI Interview Knowledge RAG",
    page_icon="🧠",
    layout="wide",
)

WORKFLOW_HELP = """
**LangGraph path**

`START → Query Rewriter → Retriever → Re-ranker`
→ if relevance is low and retries remain: **Query Rewriter** again
→ **Answer Generator → Citation Adder → END**
"""


def _api_key_ok() -> bool:
    key = get_settings().gemini_api_key
    return bool(key) and not key.startswith("your_")


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def render_sidebar() -> tuple[float, int]:
    settings = get_settings()
    with st.sidebar:
        st.title("Settings")
        st.markdown("AI Interview Knowledge RAG · Gemini + Chroma + LangGraph")

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(settings.default_temperature),
            step=0.05,
            help="Lower is more deterministic — better for interview facts.",
        )
        k = st.slider(
            "Retrieved documents (k)",
            min_value=2,
            max_value=8,
            value=int(settings.default_k),
        )

        st.markdown("---")
        st.markdown(WORKFLOW_HELP)
        st.caption(f"Collection: `{settings.chroma_collection_name}`")
        st.caption(f"Chat model: `{settings.gemini_chat_model}`")

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_result = None
            st.rerun()

        st.markdown("---")
        if st.button("Rebuild knowledge base", use_container_width=True):
            from ingest import ingest

            with st.spinner("Re-ingesting documents into ChromaDB..."):
                n = ingest()
            st.success(f"Stored {n} chunks.")
    return temperature, k


def main() -> None:
    init_state()
    temperature, k = render_sidebar()

    st.title("AI Interview Knowledge RAG System")
    st.caption("Ask about RAG, LangGraph, transformers, fine-tuning, and other interview topics.")

    if not _api_key_ok():
        st.error(
            "Add your Gemini API key first: copy `.env.example` to `.env` and set `GEMINI_API_KEY`."
        )
        st.stop()

    try:
        n_chunks = ensure_knowledge_base()
    except Exception as exc:
        st.error(f"Could not load the knowledge base: {exc}")
        st.stop()

    st.success(f"Knowledge base ready · {n_chunks} chunks in ChromaDB", icon="✅")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Retrieved sources"):
                    for i, doc in enumerate(msg["sources"], start=1):
                        st.markdown(
                            f"**[{i}] {doc['title']}**  \n"
                            f"`{doc['source']}` · similarity={doc.get('score')} · "
                            f"relevance={doc.get('relevance')}\n\n"
                            f"{doc['content'][:500]}..."
                        )
            if msg["role"] == "assistant" and msg.get("steps"):
                with st.expander("LangGraph workflow steps"):
                    for step in msg["steps"]:
                        st.write(f"- {step}")

    prompt = st.chat_input("Ask an AI/ML interview question...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Running LangGraph workflow..."):
            try:
                result = run_rag(prompt, k=k, temperature=temperature)
            except Exception as exc:
                st.error(f"Graph execution failed: {exc}")
                return

        answer = result["answer"]
        st.markdown(answer)
        with st.expander("Retrieved sources", expanded=True):
            docs = result.get("documents") or []
            if not docs:
                st.write("No documents retrieved.")
            for i, doc in enumerate(docs, start=1):
                st.markdown(
                    f"**[{i}] {doc['title']}**  \n"
                    f"`{doc['source']}` · similarity={doc.get('score')} · "
                    f"relevance={doc.get('relevance')}\n\n"
                    f"{doc['content'][:700]}"
                )
        with st.expander("LangGraph workflow steps", expanded=True):
            for step in result.get("steps") or []:
                st.write(f"- {step}")
            st.caption(f"Rewritten query: {result.get('rewritten_query')}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": result.get("documents") or [],
            "steps": result.get("steps") or [],
        }
    )
    st.session_state.last_result = result


main()
