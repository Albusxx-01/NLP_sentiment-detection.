# NLP Transformer — Sarcasm Detection

Sarcasm detection is the task of classifying a piece of text — here a news headline — as *sarcastic* or *not sarcastic*. Sarcasm relies on context, tone, and subtle wording that literal keyword matching misses. By fine-tuning a pretrained transformer (DistilBERT), this model learns those contextual subtleties, relates each headline to the real world the article refers to, and outputs an accurate, succinct verdict of intent. It is a task that, until recent years, only humans could do almost effortlessly.

## Problem Statement

Sarcasm detection is an interesting problem because it sits at the intersection of natural language processing and pragmatics — a news headline can be grammatically correct and factually accurate yet mean the opposite of what it says. In this case study we follow the **fine-tuning** paradigm: take a pretrained transformer (DistilBERT), add a classification head, and fine-tune it on the *Sarcasm Headlines Dataset*. The model takes a single news headline as input and outputs the predicted label (`sarcastic` / `not_sarcastic`) along with a confidence score, which can be read out directly in the web app.

## Dependencies

- Python 3.13
- PyTorch 2.13 (CPU; GPU optional)
- HuggingFace transformers / tokenizers
- gtts n/a — results shown as text + confidence in a Streamlit demo / FastAPI

## Business Objectives and Constraints

- Predict the correct label for each input headline.
- An incorrect label could leave a negative impression on the user and erode trust in the product.
- No strict latency constraints — a streamed/CPU response in a few hundred ms is acceptable.
- Objective, reproducible target: reach **F1 ≥ 0.85** on a held-out test set.

## Data Overview

The **Sarcasm Headlines Dataset v2** (Rishabh Misra) contains **28,619 headlines** collected from two news sites — *TheOnion* (sarcastic) and *HuffPost* (real news) — each labeled with `is_sarcastic` (0/1). Headlines are short, cleaned, real-world samples that capture everyday ironic news language. The dataset is stored as JSON-Lines (`Sarcasm_Headlines_Dataset.json`, ~6 MB).

- After cleaning/dedup: ~28.6k unique headlines, **~47.5% sarcastic** (near-balanced)
- Stratified split: **train / val / test = 5000 / 1000 / 1000**, each preserving the sarcastic share

**Sources:**
- https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection (the original `Sarcasm-Headlines-Dataset` repo is 404)
- https://rishabhmisra.github.io/

## Mapping the Real-World Problem to a Deep Learning Problem

We treat this as **binary text classification**. Each headline is tokenized with the DistilBERT subword tokenizer (`max_length=128`, padding/truncation, attention masks) and fed to `AutoModelForSequenceClassification`, a pretrained transformer with a single linear classification head. During training we minimize cross-entropy loss with AdamW, linear warmup + decay, and save the best checkpoint by validation F1. The approach mirrors the modern transformer fine-tuning recipe introduced by *Attention Is All You Need* (Vaswani et al., 2017) and popularized by Google's BERT / HuggingFace DistilBERT.

## Key Performance Indicator (KPI)

**F1 score** is the primary metric (harmonic mean of precision and recall) — well suited to a near-balanced but noisy text class. Given the model's predicted label vs. the ground-truth label over the test set:

- F1 close to **1** means predictions are precise and complete; the closer to 1, the better.
- Accuracy, precision, recall, and the confusion matrix are reported alongside F1.
- Target: **F1 ≥ 0.85** on the held-out test set — higher is better.

Reference: https://en.wikipedia.org/wiki/F-score

## Results

Fine-tuned DistilBERT (2 epochs, AdamW, linear warmup + decay) on 5,000 train / 1,000 val / 1,000 test headlines.

| Metric | Value |
|--------|-------|
| Accuracy | **0.8770** |
| Precision | 0.9000 |
| Recall | 0.8337 |
| **F1** | **0.8656** ✅ (target ≥ 0.85) |

- Misclassified: 123 / 1000 (12.30%)
- Artifacts: `reports/evaluation_metrics.json`, `reports/confusion_matrix.png`, `reports/error_analysis.csv`, checkpoint in `models/checkpoint/`

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

- Prepare data: `python -m src.data.prepare`
- Train (local CPU legacy; pref. use the Colab notebook): `python -u -m src.model.train`
- Train on GPU: open `notebooks/sarcasm_finetune_colab.ipynb` in Google Colab
- Evaluate: `python -m src.model.evaluate`
- Run demo app: `streamlit run src/app/demo.py`
- Run API: `uvicorn src.app.api:app --reload`
- Run tests: `python -m pytest tests/ -q`

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

## References

- HuggingFace fine-tuning tutorial: https://huggingface.co/docs/transformers/training
- BERT: Pre-training of Deep Bidirectional Transformers: https://arxiv.org/pdf/1810.04805.pdf
- DistilBERT, a distilled version of BERT: https://arxiv.org/pdf/1910.01108.pdf
- Attention Is All You Need: https://arxiv.org/pdf/1706.03762.pdf
- CS231n: Convolutional Neural Networks for Visual Recognition: https://www.youtube.com/watch?v=NfnWJUyUJYU
- Seq2seq models (Andrew Ng): https://www.youtube.com/watch?v=Q8ys8YnDRXM
- The Sarcasm Headlines Dataset: https://rishabhmisra.github.io/publications/
- F-score: https://en.wikipedia.org/wiki/F-score
- [PROGRESS](PROGRESS.md) — live status + resume guide