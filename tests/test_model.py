"""Smoke tests for the fine-tuned model (require a trained checkpoint)."""

import pytest
import torch

from src.config import ROOT, load_config

CFG = load_config()
CHECKPOINT_DIR = ROOT / CFG["model"]["checkpoint_dir"]

requires_model = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "config.json").exists(),
    reason="No trained checkpoint found; run training first",
)


@requires_model
def test_model_loads_and_predicts() -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR)
    model.eval()

    headlines = [
        "A completely normal news headline",
        "Man shoots neighbor in dispute over barking dog",
    ]
    inputs = tokenizer(
        headlines,
        padding=True,
        truncation=True,
        max_length=CFG["tokenizer"]["max_length"],
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    assert tuple(logits.shape) == (len(headlines), CFG["model"]["num_labels"])

    probs = torch.softmax(logits, dim=-1)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)
    labels = logits.argmax(dim=-1).tolist()
    assert all(label in {0, 1} for label in labels)


@requires_model
def test_config_matches_checkpoint() -> None:
    import json

    with open(CHECKPOINT_DIR / "config.json", encoding="utf-8") as f:
        ckpt_cfg = json.load(f)

    assert ckpt_cfg.get("model_type") == "distilbert"
    assert "DistilBertForSequenceClassification" in ckpt_cfg.get("architectures", [])
