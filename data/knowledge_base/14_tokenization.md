# Tokenization and Context Windows

Tokenization splits raw text into subword units (tokens) the model was trained on. Algorithms include BPE, WordPiece, and SentencePiece. One English word is often 1–2 tokens; rare terms and code can use more.

## Why interviews ask this
Pricing, context limits, and RAG chunk sizes are token-based, not word-based. A "8k context" model cannot take an unbounded PDF. You budget tokens for system prompt + retrieved chunks + user question + generated answer.

## Chunking in token space
Character-based splitters are an approximation. Production systems often split with a token counter so k chunks never overflow the window. Leave headroom for the answer.

## Special cases
- Leading spaces and punctuation can be their own tokens.
- Multilingual text tokenizes less efficiently on English-centric vocabs.
- Identical strings can have different token counts across model families.

## Practical RAG implication
Retrieve fewer, better chunks rather than stuffing the window. Re-ranking exists partly to save context budget for the most relevant evidence.

## Interview one-liner
Models read tokens, not words; context windows and costs are in tokens, so RAG chunking and prompt size must be planned in that unit.
