"""Evaluate the fine-tuned model on the test split and write metrics + plots."""

import json
import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import HF_CACHE_DIR, ROOT, load_config
from src.data.tokenize import build_loaders

logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))


@torch.no_grad()
def predict(model, loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[list[int], list[int], list[float]]:
    model.eval()
    preds: list[int] = []
    trues: list[int] = []
    probs: list[float] = []
    for batch in loader:
        input_ids, attention_mask, labels = (t.to(device) for t in batch)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        p = torch.softmax(logits, dim=-1)[:, 1]
        preds.extend(logits.argmax(dim=-1).tolist())
        trues.extend(labels.tolist())
        probs.extend(p.tolist())
    return preds, trues, probs


def plot_confusion_matrix(cm: np.ndarray, classes: list[str], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    model_cfg = cfg["model"]
    ds_cfg = cfg["dataset"]
    eval_cfg = cfg["evaluation"]

    checkpoint_dir = ROOT / model_cfg["checkpoint_dir"]
    if not (checkpoint_dir / "config.json").exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_dir}. Run training first.")
    report_dir = Path(eval_cfg["report_dir"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    loaders, _ = build_loaders(cfg)
    test_loader = loaders["test"]
    logger.info("Loaded test split: %d batches", len(test_loader))

    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir,
        cache_dir=str(HF_CACHE_DIR),
    )
    model.to(device)

    preds, trues, probs = predict(model, test_loader, device)
    accuracy = accuracy_score(trues, preds)
    precision = precision_score(trues, preds)
    recall = recall_score(trues, preds)
    f1 = f1_score(trues, preds)
    cm = confusion_matrix(trues, preds)

    logger.info("Test accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f", accuracy, precision, recall, f1)
    logger.info("\n%s", classification_report(trues, preds, target_names=list(ds_cfg["classes"].values()), digits=4))

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "test_examples": len(trues),
        "target_f1": eval_cfg["target_f1"],
        "confusion_matrix": cm.tolist(),
    }
    metrics_path = report_dir / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Wrote metrics to %s", metrics_path)

    plot_confusion_matrix(cm, list(ds_cfg["classes"].values()), report_dir / "confusion_matrix.png")
    logger.info("Wrote confusion matrix plot to %s", report_dir / "confusion_matrix.png")

    df = pd.read_parquet(ROOT / cfg["data"]["processed_path"])
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    mask = np.asarray(trues) != np.asarray(preds)
    errors = test_df.loc[mask].copy()
    errors["predicted"] = [preds[i] for i in errors.index]
    errors["actual"] = [trues[i] for i in errors.index]
    errors["prob_sarcastic"] = [probs[i] for i in errors.index]
    errors_path = report_dir / "error_analysis.csv"
    errors.to_csv(errors_path, index=False)
    logger.info("Wrote %d misclassified examples to %s", len(errors), errors_path)

    n_errors = len(preds) - sum(1 for p, t in zip(preds, trues) if p == t)
    logger.info(
        "Misclassified: %d / %d (%.2f%%). Stay within target F1=%.2f -> %s",
        n_errors,
        len(preds),
        100.0 * n_errors / len(preds),
        eval_cfg["target_f1"],
        "PASS" if f1 >= eval_cfg["target_f1"] else "BELOW TARGET",
    )


if __name__ == "__main__":
    main()
