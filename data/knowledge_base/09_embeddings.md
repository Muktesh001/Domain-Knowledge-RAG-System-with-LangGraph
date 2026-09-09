# Embeddings

Embeddings are dense numeric vectors that represent text (or other modalities) so that semantic similarity becomes geometric closeness. "King - man + woman ≈ queen" is the classic intuition; modern sentence embeddings capture paraphrase similarity for retrieval.

## How RAG uses embeddings
Each chunk is mapped to a vector. The query is mapped with the **same** model. Nearest neighbors become the LLM context. Asymmetric models sometimes embed queries and documents differently (e.g. instruction prefixes).

## Properties interviewers expect
- **Dimensionality**: 256–3072 is common; higher is not automatically better.
- **Normalization**: cosine similarity assumes comparable vector norms.
- **Domain shift**: legal or medical text may need specialized models.
- **Multilingual**: not all embedding models handle all languages equally.

## Google text-embedding models
Gemini-era Google embeddings (for example `text-embedding-004`) are used via API. They are convenient when the chat model is also Gemini, but you can mix providers if you keep one embedding space per collection.

## Chunking interaction
Embeddings encode a window of text. If a chunk mixes unrelated topics, similarity scores become noisy. Cleaning (unicode normalize, drop boilerplate) before embedding measurably helps retrieval.

## Interview one-liner
Embeddings turn text into vectors so related meanings lie nearby; RAG quality depends as much on the embedding model and chunking as on the LLM.
