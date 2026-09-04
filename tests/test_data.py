"""Smoke tests for the data pipeline."""

import pandas as pd
import pytest

from src.config import ROOT, load_config
from src.data.prepare import clean, label_distribution, load_raw, stratify
from src.data.tokenize import build_tokenizer, tokenize_split

CFG = load_config()
DATA_CFG = CFG["data"]
DS_CFG = CFG["dataset"]
TOK_CFG = CFG["tokenizer"]


@pytest.fixture(scope="session")
def processed() -> pd.DataFrame:
    return pd.read_parquet(ROOT / DATA_CFG["processed_path"])


def test_raw_dataset_exists_and_sizeable() -> None:
    df = load_raw(ROOT / DATA_CFG["raw_path"])
    assert len(df) >= 28_000
    assert "headline" in df.columns and "is_sarcastic" in df.columns


def test_clean_removes_duplicates() -> None:
    df = pd.DataFrame(
        {
            "headline": ["a b", "a b", "", "c d"],
            "is_sarcastic": [1, 0, 1, 0],
        }
    )
    out = clean(df, "headline", "is_sarcastic")
    assert len(out) == 2
    assert set(out["headline"]) == {"a b", "c d"}


def test_stratify_sizes_and_balance() -> None:
    df = load_raw(ROOT / DATA_CFG["raw_path"])
    df = clean(df, DS_CFG["text_col"], DS_CFG["label_col"])
    train, val, test = stratify(
        df,
        DS_CFG["label_col"],
        DATA_CFG["train_subset"],
        DATA_CFG["val_subset"],
        DATA_CFG["test_subset"],
        CFG["training"]["seed"],
    )
    assert len(train) == DATA_CFG["train_subset"]
    assert len(val) == DATA_CFG["val_subset"]
    assert len(test) == DATA_CFG["test_subset"]
    for part in (train, val, test):
        share = part[DS_CFG["label_col"]].mean()
        assert 0.40 <= share <= 0.55


def test_processed_has_expected_shape_and_splits(processed: pd.DataFrame) -> None:
    assert set(processed["split"].unique()) == {"train", "val", "test"}
    counts = processed["split"].value_counts()
    assert counts["train"] == DATA_CFG["train_subset"]
    assert counts["val"] == DATA_CFG["val_subset"]
    assert counts["test"] == DATA_CFG["test_subset"]
    assert processed[DS_CFG["text_col"]].isna().sum() == 0


def test_tokenizer_and_shapes(processed: pd.DataFrame) -> None:
    tokenizer = build_tokenizer(TOK_CFG["name"])
    part = processed[processed["split"] == "test"].head(8)
    ds = tokenize_split(
        part,
        tokenizer,
        DS_CFG["text_col"],
        DS_CFG["label_col"],
        TOK_CFG["max_length"],
    )
    input_ids, attention_mask, labels = ds[0]
    assert tuple(input_ids.shape) == (TOK_CFG["max_length"],)
    assert tuple(attention_mask.shape) == (TOK_CFG["max_length"],)
    assert int(input_ids.max()) < tokenizer.vocab_size
    assert int(labels) in {0, 1}


def test_label_distribution_stable(processed: pd.DataFrame) -> None:
    dist = label_distribution(processed, DS_CFG["label_col"])
    assert abs(float(dist.get(1, 0.0)) - 0.475) < 0.05
    assert abs(float(dist.get(0, 0.0)) - 0.525) < 0.05