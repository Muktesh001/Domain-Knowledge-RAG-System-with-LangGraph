# BERT vs GPT Style Models

BERT (Bidirectional Encoder Representations from Transformers) is an encoder-only model pretrained with masked language modeling and next-sentence prediction (original paper). GPT-style models are decoder-only, pretrained with next-token prediction under a causal mask.

## Bidirectional vs causal
BERT sees left and right context, which is excellent for classification, NER, and embedding-style tasks. GPT cannot see future tokens during training, which matches open-ended generation.

## Typical use
- **BERT family**: classification, retrieval encoders, token tagging. Sentence-BERT and similar models produce high-quality embeddings for RAG.
- **GPT family**: chat, code, agents, long-form generation.

## Fine-tuning differences
BERT often adds a small head and fine-tunes for a label. GPT-style models are instruction-tuned and aligned for conversation; you prompt them or apply LoRA rather than training a tiny classifier head only.

## For RAG interviews
You might use a BERT-like encoder as the embedding model and a GPT-like decoder as the generator. They live in different stages of the pipeline.

## Interview one-liner
BERT is bidirectional and encoder-oriented (understanding); GPT is causal and decoder-oriented (generation). RAG often uses an encoder for retrieval and a decoder LLM for answering.
