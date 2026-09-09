# Fine-Tuning Large Language Models

Fine-tuning continues training a pretrained model on task- or domain-specific data so its weights (or a subset) adapt. It is complementary to RAG: fine-tuning changes behavior and style; RAG injects fresh facts.

## Common recipes
- **Full fine-tuning**: update all weights. Highest fidelity, expensive, high risk of catastrophic forgetting.
- **LoRA / QLoRA**: freeze the base model, train low-rank adapters. Popular for domain chat and instruction following on modest GPUs.
- **Instruction tuning**: SFT on (instruction, response) pairs.
- **Preference tuning**: RLHF, DPO, or similar methods to align with human rankings.

## When to fine-tune vs RAG
Fine-tune when you need a consistent voice, tool-calling format, or domain jargon the base model mishandles. Use RAG when facts change weekly or you need citations from private corpora. Many systems do both: a lightly adapted model plus retrieval.

## Data quality
Label noise, duplicated web text, and leakage from eval sets ruin fine-tunes. Deduplicate, clean, and hold out a real test set. For interviews, mention evaluation: win rates, task accuracy, and regression on general benchmarks.

## Risks
Memorization of private data, reward hacking, and over-narrowing the model. Version datasets and adapters like any other artifact.

## Interview one-liner
Fine-tuning adapts model weights (often via LoRA) for style and skills; it is not a substitute for retrieval when knowledge must stay current and citable.
