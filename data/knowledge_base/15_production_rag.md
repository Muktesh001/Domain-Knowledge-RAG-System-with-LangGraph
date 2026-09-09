# RAG Production Checklist and Re-ranking

## Ingestion hygiene
Clean text before embedding: unicode normalize, strip control characters, drop HTML, collapse whitespace. Remove near-duplicate chunks. Store metadata: title, source file, section heading, timestamps.

## Query-side improvements
- **Query rewriting**: expand acronyms (RAG, LoRA) and add missing terms.
- **HyDE**: generate a hypothetical answer and embed that (advanced).
- **Multi-query**: rewrite into several queries and merge hits (dedupe).

## Re-ranking
First-stage ANN is fast but coarse. A **re-ranker** (cross-encoder or an LLM relevance grader) scores each (query, chunk) pair and keeps the top n. This is one of the highest-ROI upgrades after decent chunking.

## Conditional graphs
If the top relevance score is below a threshold, LangGraph should rewrite the query and retrieve again, with a max-attempt cap to avoid infinite loops. If still poor, the answer node should abstain rather than hallucinate.

## Observability
Log rewritten queries, retrieved IDs, grader scores, latency, and token usage. Trace each graph node. This is what "production-ready" means beyond a demo notebook.

## Interview one-liner
Production RAG is a pipeline: clean ingest, query rewrite, retrieve, re-rank, generate, cite, and gate on relevance—with traces so you can debug misses.
