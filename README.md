# NLP Transformer — Sarcasm Detection

Fine-tune a pretrained transformer (DistilBERT) to classify news headlines as sarcastic or not.

## Status

| Stage | Description                | Status |
|-------|----------------------------|--------|
| 1     | Scaffolding                | —      |
| 2     | Data pipeline              | —      |
| 3     | Training                   | —      |
| 4     | Evaluation                 | —      |
| 5     | Serving                    | —      |
| 6     | Finalize (tests, lint)     | —      |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

- Download & prepare data: `python -m src.data.prepare`
- Train: `python -m src.model.train`
- Evaluate: `python -m src.model.evaluate`
- Run demo app: `streamlit run src/app/demo.py`
- Run API: `uvicorn src.app.api:app --reload`

## Project Structure

```
03-nlp-transformer/
├── pre-build/          # PRD, architecture, prerequisites
├── data/               # raw + processed (gitignored)
├── notebooks/          # EDA & experiments
├── src/
│   ├── data/           # load, clean, tokenize, split
│   ├── model/          # fine-tune, evaluate
│   └── app/            # Streamlit demo + FastAPI
├── models/             # fine-tuned checkpoints (gitignored)
├── reports/            # metrics & plots (gitignored)
├── tests/
├── config.yaml         # all tunable parameters
└── requirements.txt
```

## Documentation

- [PROGRESS](PROGRESS.md) — live status + resume guide
- [PRD](pre-build/prd.md)
- [Architecture](pre-build/architecture.md)
- [Prerequisites](pre-build/prerequisites.md)# NLP_sentiment-detection.
