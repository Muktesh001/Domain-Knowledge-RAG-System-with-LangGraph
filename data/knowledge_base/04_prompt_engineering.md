# Prompt Engineering

Prompt engineering is the practice of designing instructions, context, and examples so a language model produces reliable outputs. In interviews, it is treated as a product skill, not magic words.

## Techniques that actually matter
- **System prompts**: set role, constraints, and output format (e.g. JSON only).
- **Few-shot examples**: show input-output pairs so the model copies structure and tone.
- **Chain of thought**: ask for stepwise reasoning when the task is multi-hop. For RAG answers, prefer "use only the context" over unconstrained reasoning.
- **Delimiter and structure**: labeled sections (Question, Context, Rules) reduce instruction bleed.
- **Output schemas**: JSON mode or parsers make downstream code robust.

## RAG-specific prompting
The generator prompt should require grounding: "Answer using ONLY the provided context. If missing, say you don't know." Ask for citations by chunk id. Separate rewrite, grade, and answer prompts so each node has one job.

## Failure modes
Vague prompts, mixed tasks in one prompt, and missing negative constraints ("do not invent APIs") cause drift. Overly long prompts bury the question. Evaluate prompts with a held-out question set, not vibes.

## Interview one-liner
Prompt engineering is specifying role, constraints, examples, and format so the model’s behavior is testable and grounded—especially important in RAG where the model must not override retrieved evidence.
