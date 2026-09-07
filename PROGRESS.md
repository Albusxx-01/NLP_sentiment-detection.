# Project Progress — NLP Transformer (Sarcasm Detection)

> Live status + resume guide. All work happens inside `03-nlp-transformer/` only.

Last updated: 2026-09-07 (Session 4 — all six stages DONE, commit+push pending)

---

## Git — Branch & Commit Record

- Repo: nested repo inside the project folder. Branch `main`, remote `origin/main` exists.
  NOT pushed (ahead of origin by several commits).
- Strategy: one feature branch per stage, merged into `main` with `--no-ff`.
- `data/`, `models/`, `reports/` are gitignored — only `.gitkeep` is tracked.

### `main` history (top → oldest)

```
c7eaaea Merge feature/evaluation: metrics, confusion matrix, error analysis
|  2d7479f feat(model): add evaluation with metrics and confusion matrix
500bea3 Merge feature/training: fine-tuning script + progress tracker
|  b2a84b4 docs: add project progress tracker
|  6239db9 feat(model): add fine-tuning training script
461b84b Merge feature/data-pipeline: data prep, tokenization, smoke tests
|  7212336 test(data): add smoke tests for data pipeline
|  a6dee35 feat(data): tokenize headlines and build dataloaders
|  18f20a0 feat(data): add config loader and dataset prep pipeline
8da5e2d Merge feature/scaffolding: docs + project scaffolding
|  9b46baf chore: add project scaffolding
|  144d985 docs: add PRD, architecture, prerequisites
a8de62b first commit
```

### Branches

| Branch | Status | Merge commit |
|--------|--------|--------------|
| `feature/scaffolding` | ✅ merged | `8da5e2d` |
| `feature/data-pipeline` | ✅ merged | `461b84b` |
| `feature/training` | ✅ merged | `500bea3` |
| `feature/evaluation` | ✅ merged | `c7eaaea` |
| `feature/serving` | 📝 code written, not committed | — |
| `feature/finalize` | 📝 done, not committed | — |

### Uncommitted files (not yet on any branch)

| File | Status | Stage |
|------|--------|-------|
| `PROGRESS.md` | modified (this update) | docs |
| `src/app/demo.py` | new, verified | Stage 5 — Serving (Streamlit) |
| `src/app/api.py` | new, verified | Stage 5 — Serving (FastAPI) |
| `tests/test_app.py` | new, passing | Stage 5 — Serving tests |
| `tests/test_model.py` | new, passing | Stage 6 — Model tests |

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
- Tests: 11 passed (6 data + 3 app + 2 model), none skipped

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

### [x] Stage 3 — Training (DONE — trained on Colab GPU)

- Local CPU segfault was **never resolved** (torch CPU + Python 3.13 OpenMP clash).
- **Resolution: trained in Google Colab on a GPU** via a self-contained notebook:
  `notebooks/sarcasm_finetune_colab.ipynb` (downloads its own data, splits, trains,
  evaluates, saves checkpoint). No manual upload needed.
  (Notebook iterations fixed for transformers 5.x: `warmup_steps`,
  `eval_strategy`, removed `logging_dir` + `Trainer(tokenizer=...)`.)
- **Checkpoint downloaded & placed at `models/checkpoint/`**:
  `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`.
  Source archive `models/checkpoint.zip` (1.5 GB) also on disk — can be deleted.
- `reports/training_log.json` was NOT produced (Colab trainer tracks its own history)
  — not critical; Colab ran 2 epochs, 314 steps.
- Old diagnostics in `reports/` (`diag*.py/.log`, `training_full.log`,
  `training_console.log`) are stale — no longer relevant, optional cleanup.

### [x] Stage 4 — Evaluation (DONE — PASS)

- Ran `python -m src.model.evaluate` on the downloaded checkpoint.
- **Results (test set, 1000 examples):**
  - Accuracy **0.8770**, Precision **0.9000**, Recall **0.8337**, **F1 0.8656**
  - Target F1 ≥ 0.85 → **PASS**
  - Confusion matrix: TP=396, FP=44, FN=79, TN=481 (123/1000 misclassified)
- Outputs in `reports/`: `evaluation_metrics.json`, `confusion_matrix.png`,
  `error_analysis.csv` (123 misclassifications).
- **Fixed a bug in `evaluate.py`:** `test_df[trues != preds]` raised
  `KeyError: True` (Python lists compared inside a pandas `[]` indexer). Replaced
  with a numpy boolean mask: `mask = np.asarray(trues) != np.asarray(preds)`.
- Tests: full suite **11 passed** (was 7 passed / 2 skipped before the checkpoint existed).

### [x] Stage 5 — Serving (DONE — verified)

- `src/app/demo.py` — Streamlit demo (verified booting headless, serves on localhost)
- `src/app/api.py` — FastAPI endpoint (verified via TestClient)
  - `GET /health` → `{status: ok, checkpoint: checkpoint}`
  - `POST /predict` → `{label, label_name, confidence}`
  - Sample predictions: "Man shoots neighbor…" → sarcastic 0.9277; "Study finds
    listening to music…" → sarcastic 0.8375; "Boeing says new plane…" → not_sarcastic 0.517
- **Fixed a bug in both apps:** `classes` config keys are ints (`{0:.., 1:..}`), but code
  used `names[str(label)]` → `KeyError: '1'`. Changed to `names[label]`.
- All 3 `tests/test_app.py` pass (previously skipped without checkpoint).
- Commands: `streamlit run src/app/demo.py`, `uvicorn src.app.api:app --reload`.

### [x] Stage 6 — Finalize (DONE — lint skipped by user choice)

- **Linting skipped** — user chose to skip ruff/black/flake8 (none installed, not added
  to requirements.txt). No lint config added.
- **Cleanup done:**
  - Deleted stale diagnostics: `reports/diag*.py/.log`, `training_console.log`,
    `training_full.log`, `tokenize_rerun.log`, `streamlit_boot.log`
  - Deleted `models/checkpoint.zip` (1.5 GB, already extracted)
  - `reports/` now holds only real artifacts: `evaluation_metrics.json`,
    `confusion_matrix.png`, `error_analysis.csv`
- **Tests hardened:** added `tests/test_model.py` (2 smoke tests):
  - `test_model_loads_and_predicts` — loads the checkpoint, runs a forward pass on 2
    headlines, asserts logits shape `(2, num_labels)`, probabilities in [0,1],
    labels ∈ {0,1}
  - `test_config_matches_checkpoint` — asserts checkpoint `model_type=distilbert` and
    architecture `DistilBertForSequenceClassification`
  - Full suite: **11 passed** (6 data + 3 app + 2 model), none skipped.
- `config.yaml` `training.fp16: false` refers to the legacy local CPU path; Colab
  training used fp16 on GPU (noted in the notebook). Left as-is.
- Outstanding: commit + push the many pending files when the user authorizes.

---

## Where To Start (resume here)

1. **All six stages are DONE.** Model trained on Colab GPU (F1 0.8656, PASS),
   evaluated, served (Streamlit + FastAPI verified), and finalized (cleanup done,
   11 tests pass). Checkpoint at `models/checkpoint/`.
2. **Remaining: commit + push (awaiting user authorization).**
   - Stage 5 files: `src/app/demo.py`, `src/app/api.py`, `tests/test_app.py`
   - `tests/test_model.py` (new)
   - App bug fixes, PROGRESS/README updates, Colab notebook, cleaned `reports/`
   - Personal docs (user-owned): `learning/`, `pre-build/format.md`,
     `pre-build/prerequisites.md`
3. **To run the live apps:**
   ```
   streamlit run src/app/demo.py
   uvicorn src.app.api:app --reload
   ```
   Tests: `python -m pytest tests/ -q` (11 passed)

---

## Key Commands

| Task | Command |
|------|---------|
| Prepare data | `python -m src.data.prepare` |
| Tokenize (verify loaders) | `python -m src.data.tokenize` |
| Train | `PYTHONUNBUFFERED=1 python -u -m src.model.train 2>&1 > reports/training_run.log` |
| Evaluate | `python -m src.model.evaluate` |
| Tests | `python -m pytest tests/ -q` |
| Demo | `streamlit run src/app/demo.py` |
| API | `uvicorn src.app.api:app --reload` |

**Gotchas:**
- All HF downloads cache to `models/hf_cache/` (project-local). Don't change `HF_HOME`.
- bash pipes kill/tamper with python child processes on Windows sometimes:
  prefer redirect-to-file (`> reports/x.log 2>&1`) over `| grep...`, and run long
  jobs foreground with a large timeout, or a log file + polling.
- Raw JSON is JSON-Lines (one object per line) — parse line by line, not `json.load`.
- **DO NOT reorder imports** in `train.py` back to module top — `AutoModelForSequenceClassification`
  must stay inside `main()` after `build_loaders()`.

## Files Map

- Config: `config.yaml`, `src/config.py`
- Pipeline: `src/data/prepare.py`, `src/data/tokenize.py`
- Training: `src/model/train.py` (CPU — segfaults; use Colab notebook instead)
- Colab training: `notebooks/sarcasm_finetune_colab.ipynb`
- Evaluation: `src/model/evaluate.py` (committed, fixed)
- Serving: `src/app/demo.py`, `src/app/api.py` (verified; not yet committed)
- Tests: `tests/test_data.py`, `tests/test_app.py`, `tests/test_model.py`
  (test_app + test_model not yet committed)
- Checkpoint: `models/checkpoint/` (`config.json`, `model.safetensors`, tokenizer)
- Metrics/plots: `reports/evaluation_metrics.json`, `reports/confusion_matrix.png`,
  `reports/error_analysis.csv`
- Data: `data/raw/Sarcasm_Headlines_Dataset.json` (raw, 28.6k),
  `data/processed/dataset.parquet` (7k, split)
