"""Tokenize headlines and build PyTorch train/val/test loaders."""

import logging
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from src.config import HF_CACHE_DIR, ROOT, load_config

logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))


def build_tokenizer(model_name: str) -> AutoTokenizer:
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return AutoTokenizer.from_pretrained(model_name, cache_dir=str(HF_CACHE_DIR))


def tokenize_split(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    text_col: str,
    label_col: str,
    max_length: int,
) -> TensorDataset:
    encodings = tokenizer(
        df[text_col].tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = torch.tensor(df[label_col].to_numpy(dtype="int64"), dtype=torch.long)
    return TensorDataset(encodings["input_ids"], encodings["attention_mask"], labels)


def build_loaders(cfg: dict) -> tuple[dict[str, DataLoader], AutoTokenizer]:
    data_cfg = cfg["data"]
    ds_cfg = cfg["dataset"]
    tok_cfg = cfg["tokenizer"]
    train_cfg = cfg["training"]

    df = pd.read_parquet(ROOT / data_cfg["processed_path"])
    tokenizer = build_tokenizer(tok_cfg["name"])

    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        part = df[df["split"] == split].reset_index(drop=True)
        dataset = tokenize_split(
            part,
            tokenizer,
            ds_cfg["text_col"],
            ds_cfg["label_col"],
            tok_cfg["max_length"],
        )
        batch_size = train_cfg["batch_size"] if split == "train" else train_cfg["batch_size"] * 2
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
        )
        logger.info("%s split: %d examples, %d batches", split, len(part), len(loaders[split]))

    return loaders, tokenizer


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    loaders, _ = build_loaders(cfg)

    for split, loader in loaders.items():
        input_ids, attention_mask, labels = next(iter(loader))
        logger.info(
            "%s batch shapes: input_ids=%s attention_mask=%s labels=%s",
            split,
            tuple(input_ids.shape),
            tuple(attention_mask.shape),
            tuple(labels.shape),
        )
        logger.info("%s batch sample labels: %s", split, labels.tolist())


if __name__ == "__main__":
    main()