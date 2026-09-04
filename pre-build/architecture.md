# Architecture — NLP: Fine-Tune a Transformer for Sentiment / Sarcasm Detection

> Template based on `homebase/guideline.md` §2. Sketch before implementation; present to user for approval.

## System Diagram
```
Raw Text/CSV → Cleaning (labels, sampling, dedupe)
    → Tokenization (HF AutoTokenizer, max_length, padding/truncation)
    → Dataset split (stratified train/val/test)
    → Fine-tune (DistilBERT/RoBERTa + classifier head, DataLoader, AMP)
    → Evaluate (F1, confusion matrix, error analysis)
    → Gradio/Streamlit demo or FastAPI → deployable artifact
```

## Tech Stack Selection
- **Python 3.x**
- **PyTorch** — training framework
- **HuggingFace Transformers** — pretrained models + tokenizer
- **datasets** (HF) — data loading
- **accelerate / Trainer OR custom loop** — training (choose one)
- **scikit-learn** — metrics, stratified split
- **matplotlib / seaborn** — plots
- **Gradio / Streamlit / FastAPI** — serving
> Prefer libraries already used in the existing codebase where possible.

## Project Structure
```
03-nlp-transformer/
├── pre-build/          # PRD, architecture, prerequisites, checklist
├── data/               # raw + processed (gitignored if large)
├── notebooks/          # EDA & experiments
├── src/
│   ├── data/           # load, clean, tokenize, split
│   ├── model/          # fine-tune, evaluate
│   └── app/            # Gradio/Streamlit UI or FastAPI
├── models/             # fine-tuned checkpoints (gitignored)
├── tests/
├── requirements.txt / pyproject.toml
└── README.md
```

## Data Model
- Text → label (binary or N-class); label map stored in config/JSON.
- Tokenized dataset with `input_ids`, `attention_mask`, `labels`.
- Stratified split by label; shuffle only within training.

## State Management
- Stateless inference; tokenizer + model loaded once at app startup.

## Error Handling Strategy
- Handle empty/degenerate inputs; cap sequence length; log failures during training (e.g., OOM fallback to CPU).

## Security Architecture
- Length limits and content validation on inputs; no credentials in code; HF Hub tokens (if used, e.g., for private models) via env vars.

## API Design (optional)
- `POST /predict` — `{"text": "..."}` → `{"label": "positive", "confidence": 0.92}`
- `GET /classes` — available labels

## Asset / Config Management
- Config for model name, max_length, batch size, paths, thresholds.
- Model checkpoints saved with `save_pretrained` for reproducibility.

---
> Rule: Present the proposed architecture to the user for approval before writing significant code.