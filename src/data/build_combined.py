"""Build a combined sarcasm dataset from multiple sources.

Sources:
  1. Sarcasm Headlines v2 (local raw JSONL)            -> ~28.6k headlines  (~47% sarc)
  2. iSarcasmEval SemEval-2022 Task 6 (English) tweets -> ~3.5k tweets      (~25% sarc)
  3. SARC Reddit (self-annotated Reddit, English)      -> ~30k comments     (~50% sarc)
  4. MUStARD TV-dialogue utterances                    -> ~690 utterances   (~50% sarc)

Note on iSarcasm (dmbavkar): its repo only makes tweet IDs public; the tweet text is
held privately by the authors and requires a data-sharing agreement, so it cannot be
downloaded programmatically and is not included.

Writes a unified parquet to data/processed/combined.parquet with columns:
  text, label, source, split

Run from project root:
  python -m src.data.build_combined
"""

import io
import json
import logging
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ROOT, load_config

logger = logging.getLogger(__name__)

ISARCASM_URL = (
    "https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/train/train.En.csv"
)
ISARCASM_LOCAL = ROOT / "data" / "raw" / "iSarcasmEval_train.En.csv"

MUSTRAD_URL = (
    "https://raw.githubusercontent.com/Himanshu-sudo/MUStARD-dataset/master/data/sarcasm_data.json"
)
MUSTRAD_LOCAL = ROOT / "data" / "raw" / "muSTARD_sarcasm_data.json"

SARC_HF_REPO = "marcbishara/sarcasm-on-reddit"
SARC_HF_SPLIT = "sft_train"

TEXT_COL = "text"
LABEL_COL = "label"
SOURCE_COL = "source"

SARC_CAP = 30_000  # subsample cap for the large SARC/Reddit source


def load_headlines(cfg: dict) -> pd.DataFrame:
    raw_path = ROOT / cfg["data"]["raw_path"]
    rows = []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df = df[["headline", "is_sarcastic"]].rename(
        columns={"headline": TEXT_COL, "is_sarcastic": LABEL_COL}
    )
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    df[SOURCE_COL] = "headlines_v2"
    return df


def load_isarcasm() -> pd.DataFrame:
    if not ISARCASM_LOCAL.exists():
        logger.info("Downloading iSarcasmEval train.En.csv ...")
        raw = urllib.request.urlopen(ISARCASM_URL, timeout=60).read().decode("utf-8")
        ISARCASM_LOCAL.parent.mkdir(parents=True, exist_ok=True)
        ISARCASM_LOCAL.write_text(raw, encoding="utf-8")
    df = pd.read_csv(ISARCASM_LOCAL)
    df = df[["tweet", "sarcastic"]].rename(
        columns={"tweet": TEXT_COL, "sarcastic": LABEL_COL}
    )
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    df[SOURCE_COL] = "isarcasm"
    df = df.dropna(subset=[TEXT_COL])
    return df


def load_mustard() -> pd.DataFrame:
    if not MUSTRAD_LOCAL.exists():
        logger.info("Downloading MUStARD sarcasm_data.json ...")
        raw = urllib.request.urlopen(MUSTRAD_URL, timeout=60).read()
        MUSTRAD_LOCAL.parent.mkdir(parents=True, exist_ok=True)
        MUSTRAD_LOCAL.write_bytes(raw)
    with open(MUSTRAD_LOCAL, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data.values())
    df = df[["utterance", "sarcasm"]].rename(
        columns={"utterance": TEXT_COL, "sarcasm": LABEL_COL}
    )
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.lower().map(
        {"true": 1, "false": 0}
    ).fillna(df[LABEL_COL]).astype(int)
    df[SOURCE_COL] = "mu_stard"
    return df


def load_sarc_reddit(cfg: dict) -> pd.DataFrame:
    from datasets import load_dataset

    logger.info("Loading SARC Reddit from HuggingFace (%s, %s) ...", SARC_HF_REPO, SARC_HF_SPLIT)
    ds = load_dataset(SARC_HF_REPO, split=SARC_HF_SPLIT)
    df = ds.to_pandas()
    df = df.rename(columns={"comment": TEXT_COL}).copy()
    df = df[[TEXT_COL, LABEL_COL]]
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    df[SOURCE_COL] = "sarc_reddit"

    cap = cfg.get("combined", {}).get("sarc_cap", SARC_CAP)
    if len(df) > cap:
        df = df.sample(n=cap, random_state=cfg["training"]["seed"]).reset_index(drop=True)
    logger.info("SARC Reddit after subsample: %d rows", len(df))
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df = df[df[TEXT_COL].ne("")]
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    df = df.drop_duplicates(subset=[TEXT_COL], keep="first")
    return df.reset_index(drop=True)


def stratified_split(
    df: pd.DataFrame, train_frac: float, seed: int
) -> pd.DataFrame:
    train, rest = train_test_split(
        df, test_size=1 - train_frac, stratify=df[LABEL_COL], random_state=seed
    )
    val, test = train_test_split(
        rest, test_size=0.5, stratify=rest[LABEL_COL], random_state=seed
    )
    out = pd.concat(
        [
            train.assign(split="train"),
            val.assign(split="val"),
            test.assign(split="test"),
        ],
        ignore_index=True,
    )
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    cdf = cfg.get("combined", {})

    frames = [
        load_headlines(cfg),
        load_isarcasm(),
        load_sarc_reddit(cfg),
        load_mustard(),
    ]
    combined = pd.concat(frames, ignore_index=True)
    combined = clean(combined)

    for src in combined[SOURCE_COL].unique():
        part = combined[combined[SOURCE_COL] == src]
        logger.info(
            "source=%s rows=%d sarcastic_share=%.3f",
            src,
            len(part),
            part[LABEL_COL].mean(),
        )
    logger.info("total rows=%d overall_sarcastic_share=%.3f", len(combined), combined[LABEL_COL].mean())

    train_frac = cdf.get("train_frac", 0.8)
    seed = cfg["training"]["seed"]
    out = stratified_split(combined, train_frac, seed)
    logger.info(
        "train=%d val=%d test=%d",
        (out["split"] == "train").sum(),
        (out["split"] == "val").sum(),
        (out["split"] == "test").sum(),
    )

    out_path = ROOT / cdf.get("processed_path", "data/processed/combined.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(out), out_path)


if __name__ == "__main__":
    main()
