"""Load raw JSONL, clean it, and write a stratified train/val/test parquet."""

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ROOT, load_config

logger = logging.getLogger(__name__)


def load_raw(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No JSONL rows found in {path}")
    return pd.DataFrame(rows)


def clean(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    df = df[[text_col, label_col]].copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(int)
    df = df.dropna(subset=[text_col, label_col])
    df = df[df[text_col].ne("")]
    df = df.drop_duplicates(subset=[text_col], keep="first")
    return df.reset_index(drop=True)


def stratify(
    df: pd.DataFrame,
    label_col: str,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_available = len(df)
    if train_size + val_size + test_size > total_available:
        raise ValueError(
            f"Requested {train_size + val_size + test_size} rows but only "
            f"{total_available} available after cleaning"
        )

    y = df[label_col]

    train, remainder = train_test_split(
        df, test_size=val_size + test_size, stratify=y, random_state=seed
    )
    val, test = train_test_split(
        remainder,
        test_size=test_size,
        stratify=remainder[label_col],
        random_state=seed,
    )
    if len(train) > train_size:
        train, _ = train_test_split(
            train, train_size=train_size, stratify=train[label_col], random_state=seed
        )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def label_distribution(df: pd.DataFrame, label_col: str) -> pd.Series:
    return df[label_col].value_counts(normalize=True).sort_index()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    data_cfg = cfg["data"]
    ds_cfg = cfg["dataset"]

    raw_path = ROOT / data_cfg["raw_path"]
    processed_path = ROOT / data_cfg["processed_path"]
    text_col, label_col = ds_cfg["text_col"], ds_cfg["label_col"]

    processed_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_raw(raw_path)
    logger.info("Loaded %d raw rows", len(df))
    df = clean(df, text_col, label_col)
    logger.info("After cleaning/dedupe: %d rows", len(df))
    logger.info("Global label distribution:\n%s", label_distribution(df, label_col))

    train, val, test = stratify(
        df,
        label_col,
        data_cfg["train_subset"],
        data_cfg["val_subset"],
        data_cfg["test_subset"],
        cfg["training"]["seed"],
    )

    for name, part in (("train", train), ("val", val), ("test", test)):
        logger.info(
            "%s: %d rows, sarcastic share %.3f",
            name,
            len(part),
            part[label_col].mean(),
        )

    out = pd.concat(
        [
            train.assign(split="train"),
            val.assign(split="val"),
            test.assign(split="test"),
        ],
        ignore_index=True,
    )
    out.to_parquet(processed_path, index=False)
    logger.info("Wrote %d rows to %s", len(out), processed_path)


if __name__ == "__main__":
    main()