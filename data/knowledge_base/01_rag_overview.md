# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is an architecture that combines a parametric language model with a non-parametric knowledge store. Instead of asking the LLM to recall every fact from training weights, the system first retrieves relevant documents and then conditions generation on that evidence.

## Why RAG is used in production
LLMs hallucinate, go stale after the training cutoff, and cannot privately store a company's latest docs. RAG reduces those issues by grounding answers in retrieved text. Interviewers often want you to say: RAG is a retrieval-plus-generation pattern, not a single library.

## Canonical pipeline
1. **Ingest**: load documents, clean text, chunk, embed, and persist vectors.
2. **Retrieve**: embed the user query and fetch top-k similar chunks.
3. **Augment**: place retrieved text into a prompt with instructions to stay grounded.
4. **Generate**: the LLM produces an answer, ideally with citations.

## Design choices interviewers probe
- **Chunk size vs overlap**: too small loses context; too large dilutes similarity. Overlap (10–20%) helps keep sentences intact across boundaries.
- **Embedding model**: must match at query time. Domain-specific embeddings can beat general ones.
- **Hybrid search**: dense vectors plus keyword (BM25) often outperform either alone.
- **Re-ranking**: a cross-encoder or LLM grader can reorder first-stage hits.
- **Citations**: return source IDs so users can verify claims.

## Failure modes
Poor chunking, query-document vocabulary mismatch, and stale indexes cause "retrieval miss." If retrieval is weak, rewriting the query or expanding it with synonyms often recovers. RAG does not magically make the model truthful if the retrieved text is wrong.

## Interview one-liner
RAG retrieves relevant documents at inference time and feeds them to the LLM so answers are grounded, updatable, and auditable without retraining the model.
