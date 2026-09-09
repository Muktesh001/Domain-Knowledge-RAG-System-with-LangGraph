# LangGraph

LangGraph is a library for building stateful, multi-step LLM workflows as graphs. Where a simple chain is a straight line, LangGraph models nodes (functions) and edges (control flow), including loops and conditional branches.

## Why graphs
Agentic systems need memory, retries, human-in-the-loop, and branching: "if retrieval quality is low, rewrite the query; else generate." Graphs make that explicit and inspectable.

## Building blocks
- **State**: a typed dictionary (or Pydantic model) passed between nodes. Each node returns a partial update.
- **Nodes**: Python functions that read state and write fields (rewrite, retrieve, grade, generate).
- **Edges**: always-on edges or conditional edges that route to different nodes.
- **Compile**: produces a runnable app you `invoke` or stream.
- **Checkpointers** (optional): persist state for conversation memory and resume.

## RAG + LangGraph pattern
A production RAG graph often looks like:
Query rewriter -> Retriever -> Relevance grader -> (conditional rewrite) -> Answer generator -> Citation node.

This is more robust than a single prompt because each step has a focused instruction and can fail independently.

## Interview comparison
Use LangChain for components (LLM, embeddings, Chroma). Use LangGraph when you need cycles, retry policies, or multi-actor agents. LangGraph is not a replacement for vector search; it orchestrates it.

## Interview one-liner
LangGraph lets you define an agent as a state machine: nodes do work, edges decide the next step, and the graph can loop until a quality bar is met.
