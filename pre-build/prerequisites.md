# Prerequisites — NLP: Fine-Tune a Transformer for Sentiment / Sarcasm Detection

> Template based on `homebase/guideline.md` §3. Confirm each topic before building. Research/discuss anything unfamiliar BEFORE coding.

## Domain Knowledge
- Sentiment vs. sarcasm nuance; dataset biases (emoji, hashtags, noise).

## Language Fundamentals
- Python: classes, dataclasses, typing; PyTorch tensor basics.

## Framework / Platform Idioms
- HuggingFace `AutoModelForSequenceClassification`, `AutoTokenizer`, `Trainer` API; attention masks; padding/truncation.

## Database Concepts
- Loading CSV from Kaggle; optional Parquet/HF `datasets` storage.

## Version Control
- Git basics; never commit large data files or checkpoints (`.gitignore`); clear commits.

## Environment Setup
- Python 3.x + PyTorch; Windows local; GPU optional; how to run demo (`gradio` / `uvicorn`).

## Build & Dependencies
- `requirements.txt` pinned; ensure transformers/tokenizers/torch versions are compatible.

## Testing Fundamentals
- Smoke tests: tokenizer/dataset shape assertions, evaluation on small sample, app loads model.

## Deployment Concepts
- Local serving via Gradio/FastAPI; optional HuggingFace Spaces deployment; env vars for tokens.

## Tooling Aids
- Ruff linting, Black formatting, mypy type-checking.

## Security Basics
- Input length limits; prompt/text sanitation; never log raw user text unnecessarily; keep HF tokens/keys in env vars.

---
> Rule: Research/discuss any unfamiliar prerequisite BEFORE coding. Never implement a technology you haven't verified.