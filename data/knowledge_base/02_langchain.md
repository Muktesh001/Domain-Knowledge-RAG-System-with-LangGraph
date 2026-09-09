# LangChain

LangChain is a framework for building applications around large language models. It standardizes the pieces you repeatedly need: models, prompts, output parsers, retrievers, document loaders, and tools.

## Core abstractions
- **Chat models**: wrappers around providers (Gemini, OpenAI, Anthropic) with a common `invoke` / stream interface.
- **Prompt templates**: reusable system and few-shot templates with variables.
- **Chains / LCEL**: LangChain Expression Language composes runnables with the `|` operator.
- **Retrievers**: objects that take a query and return `Document` objects.
- **Agents**: LLMs that choose tools in a loop (historically ReAct-style).

## How LangChain fits a RAG app
A typical LangChain RAG path is: loader -> splitter -> embedding -> vector store -> retriever -> prompt -> chat model. LangChain does not replace your vector database; it integrates with Chroma, FAISS, Pinecone, and others.

## Interview talking points
LangChain is useful for glue and portability. Production teams still must own evaluation, tracing, prompt versioning, and failure handling. Interviewers may ask you to contrast "a chain" (mostly linear) with "an agent" (dynamic tool use).

## Common critique
LangChain abstracts many vendors, which speeds prototypes. The tradeoff is extra layers and version churn. Many production systems keep LangChain for retrievers and models, then use LangGraph when control flow becomes stateful and branching.

## Interview one-liner
LangChain is an orchestration SDK for LLM apps: prompts, models, retrievers, and tools with a shared runnable interface.
