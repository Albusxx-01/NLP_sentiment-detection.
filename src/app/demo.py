"""Streamlit demo for sarcasm detection."""

import os
from pathlib import Path

import streamlit as st

from src.config import HF_CACHE_DIR, ROOT, load_config

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))

cfg = load_config()
checkpoint_dir = ROOT / cfg["model"]["checkpoint_dir"]

st.set_page_config(page_title="Sarcasm Detector", layout="centered")
st.title("Sarcasm Detector")

if not (checkpoint_dir / "config.json").exists():
    st.error("No trained model found. Run `python -u -m src.model.train` first.")
    st.stop()


@st.cache_resource
def load_model():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, cache_dir=str(HF_CACHE_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir, cache_dir=str(HF_CACHE_DIR)
    )
    model.eval()
    return model, tokenizer


model, tokenizer = load_model()

headline = st.text_input("Enter a news headline:", placeholder="e.g. Boeing: We're fixing the door on the plane (eventually)")
threshold = st.slider("Confidence threshold", 0.5, 0.9, 0.5, 0.05)

if st.button("Predict", type="primary") or headline:
    import torch

    inputs = tokenizer(
        headline,
        padding="max_length",
        truncation=True,
        max_length=cfg["tokenizer"]["max_length"],
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    prob = torch.softmax(logits, dim=-1)[0, 1].item()
    label = 1 if prob >= threshold else 0
    names = cfg["dataset"]["classes"]

    if prob >= threshold:
        st.success(f"**Sarcastic** — confidence {prob:.2%}")
    else:
        st.info(f"**Not sarcastic** — confidence {1 - prob:.2%}")
    st.caption(f"P(sarcastic) = {prob:.4f}  |  label = {names[label]}")

st.divider()
st.caption("Demo loads the fine-tuned DistilBERT checkpoint from `models/checkpoint/`.")
