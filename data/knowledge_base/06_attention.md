# Attention Mechanism

Attention lets a model decide which tokens matter for the current prediction. In Transformers, **scaled dot-product attention** is:

Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

Queries (Q), keys (K), and values (V) are linear projections of token representations. Dividing by sqrt(d_k) keeps dot products from exploding so softmax does not saturate.

## Multi-head attention
Multiple heads project into different subspaces so the model can track syntax in one head and coreference in another. Outputs are concatenated and projected back.

## Masking
Causal masks set future positions to -inf before softmax so the decoder cannot peek ahead. Padding masks ignore pad tokens.

## Cross-attention
In encoder-decoder models, decoder queries attend to encoder keys/values (source sequence). Decoder-only LLMs usually have only self-attention plus later tool/RAG context in the prompt.

## Interview traps
- Attention is not a knowledge store; it is a routing mechanism over the current context window.
- "Attention weight" is not a perfect explanation of why a model answered something.
- Context window limits still apply: if the document is not in the window (or retrieved context), attention cannot use it.

## Interview one-liner
Attention scores which tokens to mix into each position; multi-head attention learns several of those mixing patterns in parallel.
