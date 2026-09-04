# Project Progress — NLP Transformer (Sarcasm Detection)

> Live status + resume guide. All work happens inside `03-nlp-transformer/` only.

Last updated: 2026-09-05 (Session 1, mid-Stage 3)

---

## Git

- Nested repo inside the project folder; branch `main`; remote `origin/main` exists.
- Strategy: one feature branch per stage/feature, merged into `main` with `--no-ff`.
- Done: `feature/scaffolding` merged (`docs: add PRD, architecture, prerequisites`,
  `chore: add project scaffolding`).
- Pending: `feature/data-pipeline`, `feature/training`, `feature/evaluation`,
  `feature/serving`, `feature/finalize`.
- Note: `pytest` currently dropped from requirements.txt (re-added in
  `feature/data-pipeline` with the tests commit).
- `data/`, `models/`, `reports/` are gitignored — only `.gitkeep` is tracked.

---

## Legend

- [x] done
- [~] in progress
- [ ] not started

---

## Environment (verified)

- OS: Windows, `win32`, git-bash shell
- Python: `3.13.0` (C:\Users\satwi\AppData\Local\Programs\Python\Python313\python.exe)
- PyTorch: `2.13.0+cpu`, **CUDA: NO — CPU only**, 12 threads
- transformers `5.15.1`, datasets `5.0.1`, accelerate `1.14.0`,
  pandas `2.3.3`, scikit-learn `1.8.0`, matplotlib `3.10.8`, seaborn `0.13.2`,
  streamlit `1.57.0`, fastapi `0.136.1`, pyyaml `6.0.3`, pytest `9.1.1`
- pytest installed (user-approved) and added to `requirements.txt`

---

## Stage Status

### [x] Stage 1 — Scaffolding (DONE)

- Folder tree created: `data/raw|processed`, `notebooks/`, `src/data`, `src/model`,
  `src/app`, `models/`, `reports/`, `tests/`
- `config.yaml` — all tunables (model, tokenizer, split sizes, training, eval, app)
- `requirements.txt` — pinned to installed versions
- `.gitignore` — ignores data/, models/, reports/, secrets
- `README.md` — structure + usage commands
- Package `__init__.py` files + `.gitkeep` files

### [x] Stage 2 — Data pipeline (DONE, 6/6 tests pass)

- Dataset: **Sarcasm Headlines v2** (Rishabh Misra)
  - Original repo `rishabhmisra/Sarcasm-Headlines-Dataset` URL was **404**.
  - Correct source: `rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection`
  - File: `Sarcasm_Headlines_Dataset.json` (JSON-Lines), 28,619 rows
  - Saved to `data/raw/Sarcasm_Headlines_Dataset.json` (~6 MB)
- `src/data/prepare.py`:
  - load → clean (dedupe headlines) → label-distribution report
  - stratified train/val/test = **5000 / 1000 / 1000** (~47.5% sarcastic each)
  - writes `data/processed/dataset.parquet` (7,000 rows, `split` column)
- `src/data/tokenize.py`:
  - `AutoTokenizer.from_pretrained("distilbert-base-uncased")`
  - pre-tokenizes to `max_length=128` (padding/truncation), returns PT loaders
    (batch 16 train / 32 val+test)
  - HF cache forced inside project: `models/hf_cache/` (via `HF_HOME`)
- `src/config.py` — shared config loader + `HF_CACHE_DIR` & `ROOT` paths
- Tests: `tests/test_data.py`, **6 passed** (raw size, cleaning, stratified sizes/balance,
  parquet shape, tokenize shapes, distribution)

### [~] Stage 3 — Training (IN PROGRESS — BLOCKED, workaround known)

- `src/model/train.py` written (train loop, AdamW, linear warmup+decay schedule,
  val eval/epoch, best-checkpoint save by val F1, writes `reports/training_log.json`)
- DistilBERT weights **downloaded & cached**: `models/hf_cache` = **257 MB**
- **Blocking bug (CRITICAL, DO NOT SKIP):** segfault (exit 139) when
  `AutoModelForSequenceClassification` is imported **before** `build_loaders()` runs.
  - Root cause: torch CPU + Python 3.13 OpenMP/threading clash during tokenizer batch
    work when transformers model classes are imported first.
  - **Fix applied in `src/model/train.py`:** model-class imports moved to *inside*
    `main()`, AFTER `build_loaders()` completes. Verified pattern with
    `reports/diag5.py` (delayed import → OK) vs `reports/diag3.py`/`diag4.py`
    (early import → segfault).
  - **Verification NOT yet finished:** 30-step training check
    (`reports/diag6.py`) was aborted by user. Full training has NOT run yet.
- No checkpoint exists yet: `models/checkpoint/` empty. `reports/training_log.json`
  not written.

### [ ] Stage 4 — Evaluation (NOT STARTED)

Planned: final metrics on test set (accuracy/precision/recall/F1), confusion matrix,
error analysis, plots → `reports/`.

### [ ] Stage 5 — Serving (NOT STARTED)

Planned: Streamlit demo + FastAPI `POST /predict` → `{label, confidence}`.

### [ ] Stage 6 — Finalize (NOT STARTED)

Planned: full test suite incl. training/app smoke tests, ruff/black, README wrap-up.

---

## Where To Start (resume here)

1. **Finish Stage 3 verification** — confirm the import-order workaround holds:
   ```
   PYTHONPATH=. python -u reports/diag6.py
   ```
   (30 training steps + eval acc; should print `ALL OK`, no segfault.
   Earlier run was aborted, not failed.)
2. If diag6 passes → run full training (foreground; ~10–20 min on CPU):
   ```
   python -u -m src.model.train
   ```
   Expect: epoch logs with train_loss/val acc/f1; best checkpoint saved to
   `models/checkpoint/`; `reports/training_log.json` written.
3. **IMPORTANT: do NOT reorder imports** back to module top in `train.py`.
   `from torch import nn` / `from transformers import AutoModelForSequenceClassification`
   must stay inside `main()` after `build_loaders()`. Else segfault returns.
4. After training → Stage 4 evaluation (metrics + confusion matrix on `test` split).

---

## Key Commands

| Task | Command |
|------|---------|
| Prepare data | `python -m src.data.prepare` |
| Tokenize (verify loaders) | `python -m src.data.tokenize` |
| Train | `python -u -m src.model.train` |
| Tests | `python -m pytest tests/ -q` |
| Demo (later) | `streamlit run src/app/demo.py` |
| API (later) | `uvicorn src.app.api:app --reload` |

**Gotchas:**
- All HF downloads cache to `models/hf_cache/` (project-local). Don't change `HF_HOME`.
- bash pipes kill/tamper with python child processes on Windows sometimes:
  prefer redirect-to-file (`> reports/x.log 2>&1`) over `| grep...`, and run long
  jobs foreground with a large timeout, or a log file + polling.
- Raw JSON is JSON-Lines (one object per line) — parse line by line, not `json.load`.

## Files Map

- Config: `config.yaml`, `src/config.py`
- Pipeline: `src/data/prepare.py`, `src/data/tokenize.py`
- Training: `src/model/train.py`
- Tests: `tests/test_data.py`
- Data: `data/raw/Sarcasm_Headlines_Dataset.json` (raw, 28.6k),
  `data/processed/dataset.parquet` (7k, split)
- Diagnostics: `reports/diag*.py/.log` (segfault investigation; can delete later)