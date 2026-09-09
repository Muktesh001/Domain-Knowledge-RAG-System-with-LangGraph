# Hallucinations, Grounding, and Agents vs RAG

## Hallucinations
A hallucination is a confident statement not supported by training data or provided context. Causes include next-token sampling pressure, incomplete retrieval, and prompts that demand an answer. Mitigations: RAG with "I don't know" rules, lower temperature for factual tasks, citation checks, and tool use (calculators, search).

## Grounding
Grounding means the answer is entailed by sources the user can inspect. A grounded RAG system refuses to go beyond chunks or clearly marks speculation. Citation nodes in a LangGraph workflow exist to make grounding visible.

## Agents vs RAG
RAG is usually a retrieve-then-generate (possibly with a rewrite loop). An **agent** additionally plans and calls tools: search, SQL, code execution, calendars. An agent can contain RAG as one tool ("search the knowledge base"). Not every product needs a full agent; extra loops add latency and failure modes.

## When to choose which
- Static policy PDFs, interview FAQs, internal wikis: start with RAG + graph retries.
- Multi-step tasks ("look up tickets, then draft email, then file"): agent with tools.
- Hybrid: LangGraph routes "knowledge question" to RAG and "action request" to tools.

## Interview one-liner
RAG grounds generation in documents; agents choose tools over multiple steps. Use the simplest graph that meets the product’s reliability and latency bar.
