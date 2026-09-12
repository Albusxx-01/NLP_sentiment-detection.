"""Tests for the combined sarcasm dataset builder."""

import pandas as pd
import pytest

from src.config import ROOT, load_config
from src.data.build_combined import clean, stratified_split

CFG = load_config()
COMBINED_CFG = CFG["combined"]

TEXT = "text"
LABEL = "label"
SOURCE = "source"


@pytest.fixture(scope="session")
def combined() -> pd.DataFrame:
    return pd.read_parquet(ROOT / COMBINED_CFG["processed_path"])


def test_combined_has_expected_schema(combined: pd.DataFrame) -> None:
    assert set(combined.columns) == {TEXT, LABEL, SOURCE, "split"}
    assert combined[TEXT].isna().sum() == 0
    assert set(combined[LABEL].unique()) == {0, 1}


def test_combined_contains_both_sources(combined: pd.DataFrame) -> None:
    sources = set(combined[SOURCE].unique())
    expected = {"headlines_v2", "isarcasm", "sarc_reddit", "mu_stard"}
    assert expected.issubset(sources)


def test_combined_size_and_splits(combined: pd.DataFrame) -> None:
    assert set(combined["split"].unique()) == {"train", "val", "test"}
    counts = combined["split"].value_counts()
    assert counts["train"] > 40_000
    assert counts["val"] > 5_000
    assert counts["test"] > 5_000


def test_combined_label_balance(combined: pd.DataFrame) -> None:
    share = combined[LABEL].mean()
    assert 0.35 <= share <= 0.55


def test_clean_deduplicates_and_drops_blank() -> None:
    df = pd.DataFrame(
        {
            TEXT: ["a b", "a b", "   ", "c d"],
            LABEL: [1, 0, 1, 201.5],
            SOURCE: ["s", "s", "s", "s"],
        }
    )
    out = clean(df)
    assert len(out) == 2
    assert set(out[TEXT]) == {"a b", "c d"}


def test_stratified_split_preserves_share() -> None:
    rng = pd.DataFrame(
        {TEXT: [f"t{i}" for i in range(200)], LABEL: [1, 0] * 100, SOURCE: ["s"] * 200}
    )
    out = stratified_split(rng, 0.8, 42)
    counts = out["split"].value_counts()
    assert counts["train"] == 160
    assert counts["val"] == 20
    assert counts["test"] == 20
    for part_name in ("train", "val", "test"):
        part = out[out["split"] == part_name]
        assert 0.35 <= part[LABEL].mean() <= 0.65
