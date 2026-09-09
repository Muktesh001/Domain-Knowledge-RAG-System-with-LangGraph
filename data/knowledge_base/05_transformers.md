# Transformers

The Transformer architecture, introduced in "Attention Is All You Need" (Vaswani et al., 2017), is the backbone of modern LLMs. It replaces recurrence with self-attention so tokens can interact in parallel.

## Architecture sketch
A Transformer encoder (or decoder) block typically contains:
1. Multi-head self-attention
2. Residual connection and layer normalization
3. Position-wise feed-forward network
4. Another residual + norm

Decoder-only models (GPT-style) use causal (masked) attention so token i only sees positions <= i. Encoder-only models (BERT-style) use bidirectional attention. Encoder-decoder models (T5, original Transformer) are common in sequence-to-sequence tasks.

## Why attention
Self-attention computes a weighted mix of value vectors, where weights come from query-key similarities. This captures long-range dependencies better than vanilla RNNs and trains efficiently on GPUs/TPUs.

## Positional information
Because attention is permutation-equivariant without extras, models add positional encodings (sinusoidal, learned, RoPE, ALiBi). Interviewers often ask how the model knows order.

## Scaling
LLMs scale depth, width, and data. Decoder-only Transformers dominate chat models. Efficiency variants include FlashAttention, grouped-query attention, and mixture-of-experts.

## Interview one-liner
Transformers process tokens in parallel using self-attention, enabling long-range context and the scale of today’s LLMs.
