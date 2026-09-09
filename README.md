# AI Interview Knowledge RAG System

Production-style **Retrieval-Augmented Generation** app for AI/ML interview prep.

**Stack:** LangGraph · LangChain · Google Gemini · ChromaDB (local persistent) · Streamlit · FastAPI

## Architecture

```
User question
    → Query Rewriter (Gemini + few-shot prompt)
    → Retriever (ChromaDB)
    → Re-ranker / relevance grader (Gemini)
    → (conditional) rewrite again if retrieval is weak
    → Answer Generator (grounded Gemini answer)
    → Citation / source adder
    → Response + sources in the UI
```

Knowledge lives in `data/knowledge_base/` (16 markdown documents). `ingest.py` cleans text, chunks it, embeds with Gemini, and writes a persistent Chroma collection under `chroma_db/`.

## Setup (Windows / PowerShell)

### 1. Create a virtual environment

```powershell
cd "c:\Users\mukes\Desktop\Domain Knowledge RAG System with LangGraph"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Add your Gemini API key

1. Create a key at [Google AI Studio](https://aistudio.google.com/apikey).
2. Copy the example env file and paste the key:

```powershell
copy .env.example .env
```

Edit `.env`:

```
GEMINI_API_KEY=your_real_key_here
```

### 3. Ingest the knowledge base

The Streamlit app and API will ingest automatically if Chroma is empty. To build the index yourself:

```powershell
python ingest.py
```

### 4. Run the Streamlit UI

```powershell
streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).

### 5. Optional: FastAPI

```powershell
python api.py
```

- Health: `GET http://localhost:8000/health`
- Ask: `POST http://localhost:8000/ask`

Example:

```powershell
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"What is RAG and how does LangGraph improve it?\"}"
```

## Project layout

```
├── app.py                 # Streamlit chat UI
├── api.py                 # FastAPI /ask endpoint
├── ingest.py              # Clean → chunk → embed → Chroma
├── requirements.txt
├── .env.example
├── data/knowledge_base/  # Domain documents
└── src/
    ├── config.py
    ├── cleaning.py
    ├── embeddings.py
    ├── vectorstore.py
    ├── prompts.py
    ├── bootstrap.py
    └── graph/
        ├── state.py
        ├── nodes.py
        └── workflow.py
```

## UI features

- Chat interface for interview questions
- Sidebar: temperature, `k` retrieved docs, clear chat, rebuild index
- Expandable **retrieved sources** and **LangGraph workflow steps**

## Notes

- First query after ingest calls the Gemini embedding and chat APIs (needs network).
- Re-ranking makes an extra Gemini call per retrieved chunk; keep `k` modest.
- Change models via `GEMINI_CHAT_MODEL` and `GEMINI_EMBEDDING_MODEL` in `.env`.
- If ingest fails with a 404 on embeddings, set `GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001` in `.env`.

## Sample questions

- What is RAG and why do we use it?
- How does LangGraph differ from LangChain?
- Explain self-attention and why we divide by sqrt(d_k).
- Fine-tuning vs RAG — when do you choose each?
- How would you evaluate a RAG system?
