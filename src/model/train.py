"""Fine-tune a DistilBERT sequence-classification model, then save a checkpoint."""

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.config import HF_CACHE_DIR, ROOT, load_config
from src.data.tokenize import build_loaders

logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))


def train_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    nn = torch.nn
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        input_ids, attention_mask, labels = (t.to(device) for t in batch)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        total_loss += float(loss.item())
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    preds: list[int] = []
    trues: list[int] = []
    losses: list[float] = []
    for batch in loader:
        input_ids, attention_mask, labels = (t.to(device) for t in batch)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        losses.append(float(outputs.loss.item()))
        preds.extend(outputs.logits.argmax(dim=-1).tolist())
        trues.extend(labels.tolist())
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(trues, preds)),
        "f1": float(f1_score(trues, preds)),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    eval_cfg = cfg["evaluation"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    loaders, tokenizer = build_loaders(cfg)
    train_loader, val_loader = loaders["train"], loaders["val"]

    from torch import nn
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["name"],
        num_labels=model_cfg["num_labels"],
        cache_dir=str(HF_CACHE_DIR),
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    total_steps = len(train_loader) * train_cfg["epochs"]
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: (
            step / max(1, warmup_steps) if step < warmup_steps else max(0.0, 1.0 - (step - warmup_steps) / max(1, total_steps - warmup_steps))
        ),
    )

    checkpoint_dir = ROOT / model_cfg["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_metrics: dict[str, float] = {}
    log_path = Path(eval_cfg["report_dir"]) / "training_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []

    logger.info(
        "Training: %d steps/epoch, %d epochs, lr=%.1e, warmup=%d steps",
        len(train_loader),
        train_cfg["epochs"],
        train_cfg["learning_rate"],
        warmup_steps,
    )

    for epoch in range(1, train_cfg["epochs"] + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f acc=%.4f f1=%.4f | %.1fs",
            epoch,
            train_cfg["epochs"],
            train_loss,
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["f1"],
            elapsed,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_metrics = val_metrics
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info("Saved best checkpoint to %s (val f1=%.4f)", checkpoint_dir, best_f1)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "history": history,
                "best_metrics": best_metrics,
                "device": str(device),
                "total_steps": total_steps,
            },
            f,
            indent=2,
        )
    logger.info("Training log written to %s", log_path)
    logger.info("Best val F1: %.4f (metrics %s)", best_f1, best_metrics)


if __name__ == "__main__":
    main()