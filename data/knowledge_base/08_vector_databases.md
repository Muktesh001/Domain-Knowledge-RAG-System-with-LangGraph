# Vector Databases

A vector database stores embedding vectors and supports nearest-neighbor search. In RAG, documents are embedded once; queries are embedded at runtime; the DB returns the closest chunks.

## What they provide
- Approximate nearest neighbor (ANN) indexes (HNSW, IVF, etc.)
- Metadata filtering (e.g. source == "hr_policy")
- Persistence, collections, and sometimes hybrid (BM25 + dense) search
- CRUD for incremental ingest

## ChromaDB
Chroma is a popular local/embedded vector store. It can persist to disk, which is ideal for demos and small production apps. You create a collection, add documents with embeddings and metadata, and query by vector or via a LangChain wrapper.

## Distance metrics
Cosine similarity, inner product, and L2 are common. You must use the same metric the embedding model was trained with. Never mix embedding models in one collection.

## Interview comparison
FAISS is a library (often in-process). Pinecone, Weaviate, Milvus, and pgvector target distributed production. Chroma sits in the middle: simple persistent local collections with a Python API.

## Operational concerns
Index rebuilds after embedding-model upgrades, namespace isolation per tenant, and monitoring recall@k. Persistence path and collection name should be configuration, not hard-coded secrets.

## Interview one-liner
Vector databases retrieve semantically similar chunks via ANN over embeddings; Chroma is a local persistent option well suited to RAG prototypes and single-node apps.
