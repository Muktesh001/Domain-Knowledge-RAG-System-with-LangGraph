# RAG and LLM Evaluation Metrics

You cannot ship RAG on anecdotes. Evaluation splits into retrieval quality, generation quality, and end-to-end task metrics.

## Retrieval metrics
- **Recall@k**: fraction of questions where a necessary gold chunk appears in the top k.
- **MRR / nDCG**: ranking quality when order matters.
- **Hit rate**: any relevant doc in the retrieved set.

If recall@k is low, rewriting the generator prompt will not save you.

## Generation metrics
- **Faithfulness / groundedness**: claims supported by retrieved text (LLM-as-judge or NLI).
- **Answer relevance**: addresses the actual question.
- **Correctness**: against gold answers when they exist.
- **Citation precision/recall**: cited passages actually support the sentence.

## Classic ML metrics (still asked)
For classification heads: precision, recall, F1, ROC-AUC. For ranking: precision@k. For language modeling: perplexity (not sufficient for chat quality). BLEU/ROUGE are weak for open-ended interview answers; prefer rubric scoring.

## Online metrics
Latency, token cost, thumbs-up rate, escalation to humans, and hallucination incident rate.

## Interview one-liner
Evaluate RAG in layers: retrieval recall@k first, then groundedness and answer relevance, and only then style. A fluent wrong answer is a failed system.
