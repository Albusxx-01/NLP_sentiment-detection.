"""FastAPI prediction endpoint for sarcasm detection."""

import logging
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import HF_CACHE_DIR, ROOT, load_config

logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))

cfg = load_config()
model_cfg = cfg["model"]
checkpoint_dir = ROOT / model_cfg["checkpoint_dir"]

app = FastAPI(title="Sarcasm Detection API", version="1.0.0")

_model = None
_tokenizer = None


def load_model():
    global _model, _tokenizer
    if _model is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not (checkpoint_dir / "config.json").exists():
            raise HTTPException(status_code=503, detail="No model checkpoint found. Run training first.")
        _tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, cache_dir=str(HF_CACHE_DIR))
        _model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_dir, cache_dir=str(HF_CACHE_DIR)
        )
        _model.eval()
    return _model, _tokenizer


class PredictionRequest(BaseModel):
    headline: str


class PredictionResponse(BaseModel):
    label: int
    label_name: str
    confidence: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "checkpoint": checkpoint_dir.name}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest) -> PredictionResponse:
    import torch

    model, tokenizer = load_model()
    inputs = tokenizer(
        req.headline,
        padding="max_length",
        truncation=True,
        max_length=cfg["tokenizer"]["max_length"],
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    prob_positive = torch.softmax(logits, dim=-1)[0, 1].item()
    label = 1 if prob_positive >= 0.5 else 0
    confidence = prob_positive if label == 1 else 1.0 - prob_positive
    names = cfg["dataset"]["classes"]
    return PredictionResponse(
        label=label,
        label_name=names[label],
        confidence=round(confidence, 4),
    )
