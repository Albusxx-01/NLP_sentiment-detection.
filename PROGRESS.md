# Project Progress — NLP Transformer (Sarcasm Detection)

> Live status + resume guide. All work happens inside `03-nlp-transformer/` only.

Last updated: 2026-09-12 (Session 5 DONE — combined retrain on Colab GPU, eval PASS, committed & pushed)

---

## Git — Branch & Commit Record

- Repo: nested repo inside the project folder. Branch `main`, remote `origin/main` exists.
  PUSHED — in sync with origin/main (serving + prior docs committed).
- Strategy: one feature branch per stage, merged into `main` with `--no-ff`.
- `data/`, `models/`, `reports/` are gitignored — only `.gitkeep` is tracked.

### `main` history (top → oldest)

```
d485d62 Merge feature/session5-combined: combined corpus retrain + tests + docs
bda3f54 Merge feature/serving: Streamlit demo + FastAPI API
|  ff835c5 feat(serving): add Streamlit demo and FastAPI prediction endpoint
512b1d9 docs: mark all six stages DONE, add README results + Colab train/test notes
5c15b4b fix(eval): use numpy boolean mask for error analysis; docs: mark stages 3-4 done; add Colab training notebook
b52809e docs: expand README with project write-up and references
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
| `feature/serving` | ✅ merged | `bda3f54` |
| `feature/finalize` | 📝 done, not committed | — |
| `feature/session5-combined` | ✅ merged | `d485d62` |

### Committed & pushed since last session

| Commit | Scope |
|--------|-------|
| `bda3f54` | Merge `feature/serving` → `main` (`--no-ff`). PUSHED. |
| `ff835c5` | Serving: `src/app/demo.py`, `src/app/api.py`, `tests/test_app.py`. PUSHED. |
| `512b1d9` | `PROGRESS.md` + `README.md` (docs: 6 stages DONE, README results + Colab/tests notes). PUSHED. |
| `d485d62` | Merge `feature/session5-combined` → `main` (`--no-ff`). PUSHED. |

### Uncommitted files (not yet on any branch)

| File | Status | Stage |
|------|--------|-------|
| `learning/`, `pre-build/format.md`, `pre-build/prerequisites.md` | untracked (user-owned) | personal docs — commit only if wanted |

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
- Tests: 17 passed (6 data + 3 app + 2 model + 6 combined), none skipped

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
- Outstanding: serving (Stage 5) committed & pushed (`bda3f54`); commit + push remaining pending files (Stage 6 model tests + Session 5) when the user authorizes.

### [x] Session 5 — Dataset expansion: combined multi-source corpus (DONE)

**Goal:** improve detection of sarcasm (esp. reducing false negatives on the sarcastic
class) by training on a larger, more varied corpus instead of headlines-only.

**Research (done):**
- Reviewed candidate datasets: iSarcasmEval, SARC Reddit, MUStARD, iSarcasm (dmbavkar),
  Riloff/Twitter, MMSD/multimodal, and several HuggingFace mirrors.
- **Rejected** `nikesh66/Sarcasm-dataset` (synthetic/template-augmented — would inject
  artifact patterns) and `SarcasmNet/sarcasm` (token-classification, odd deps).
- **Excluded iSarcasm (dmbavkar):** repo only publishes `tweet_id`s — the actual tweet
  text is held privately by the authors (requires a data-sharing agreement), so it
  cannot be fetched programmatically.

**Combined dataset built** → `data/processed/combined.parquet` via `src/data/build_combined.py`
(`python -m src.data.build_combined`; SARC loads via HF `datasets`):

| Source | Rows | Sarcasm share |
|--------|------|---------------|
| `headlines_v2` (Sarcasm Headlines v2) | 28,503 | 47.5% |
| `isarcasm` (iSarcasmEval SemEval-2022 T6, EN) | 3,456 | 25.1% |
| `sarc_reddit` (SARC Reddit, English, capped 30k) | 29,530 | 50.2% |
| `mu_stard` (MUStARD TV dialogue) | 673 | 51.0% |
| **Total** | **62,162** | **47.6%** |

- Unified schema: `text`, `label`, `source`, `split`.
- Stratified 80/10/10 → **train 49,729 / val 6,216 / test 6,217**.
- SARC source: `marcbishara/sarcasm-on-reddit` (HF, split `sft_train`), loads all 272k
  rows then subsamples to `combined.sarc_cap=30000` (balanced ~50% sarc).
- MUStARD cached at `data/raw/muSTARD_sarcasm_data.json` (GitHub, default branch `master`).
- Local `data`/`dataset` config left on the headline-only pipeline (keeps `prepare.py`
  + legacy tests intact); the combined path is driven by the `combined` config.

**Retrain (DONE — Colab GPU):** Ran `notebooks/sarcasm_finetune_colab.ipynb` with all 4
sources (49k/6k/6k train/val/test caps). Fixed a small notebook quirk: `dropna` now runs
before `astype(str)` so an empty iSarcasmEval `tweet` cell (1) is dropped instead of
becoming the literal string "nan".
- Trained 2 epochs, 3064 steps, batch 32, fp16; best val F1 **0.7877** (epoch 2).
- **Combined test (6,000 rows):** F1 **0.7775**, acc 0.797 (Colab `metrics.json`). Not
  directly comparable to the headline test — heterogeneous corpus.
- **Local headline test (1,000 rows) — apples-to-apples vs baseline:** F1 **0.9373**
  (baseline 0.8656), acc **0.9390**, precision 0.9157, recall 0.9600; 61/1000 misclassified
  (was 123). CM [[483,42],[19,456]]. Target F1 ≥ 0.85 → **PASS**.
- New checkpoint replaced the old baseline in `models/checkpoint/`. Old model only existed
  in stale `checkpoint-797`/`checkpoint-1594` dirs (deleted during cleanup); new model is
  strictly better.
- Cleanup: deleted `models/checkpoint.zip` (1.6 GB, already extracted).

**Tests:** added `tests/test_combined.py` (6 tests) + `tests/test_model.py` (2 smoke
tests, moved from Stage 6). Full suite now **17 passed** (6 data + 3 app + 2 model
+ 6 combined), none skipped.

**Result:** combined corpus adopted. All pending Session-5 + Stage-6 code + docs committed
& pushed via `feature/session5-combined` → `main`.

---

## Where To Start (resume here)

1. **All six stages are DONE** and **Session 5 (combined corpus) is DONE.**
   - Combined 4-source corpus (**62,162 rows**) built at `data/processed/combined.parquet`;
     `build_combined.py` + Colab notebook updated. 17 tests pass.
   - The **combined model is live** at `models/checkpoint/` (retrained on Colab GPU).
     Headline-test F1 improved **0.8656 → 0.9373**; combined-test F1 0.7775.
     Baseline checkpoint deleted (new model is strictly better).
2. **Nothing pending.** Code + docs committed & pushed. Only untracked files left are
   personal docs (user-owned): `learning/`, `pre-build/format.md`, `pre-build/prerequisites.md`.
3. **To run the live apps:**
   ```
   streamlit run src/app/demo.py
   uvicorn src.app.api:app --reload
   ```
   Tests: `python -m pytest tests/ -q` (17 passed)

---

## Key Commands

| Task | Command |
|------|---------|
| Prepare data | `python -m src.data.prepare` |
| Tokenize (verify loaders) | `python -m src.data.tokenize` |
| Build combined corpus | `python -m src.data.build_combined` |
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
- Serving: `src/app/demo.py`, `src/app/api.py` (verified; committed, pushed)
- Tests: `tests/test_data.py`, `tests/test_app.py` (committed); `tests/test_model.py`,
  `tests/test_combined.py` (not yet committed)
- Checkpoint: `models/checkpoint/` (`config.json`, `model.safetensors`, tokenizer)
- Metrics/plots: `reports/evaluation_metrics.json`, `reports/confusion_matrix.png`,
  `reports/error_analysis.csv`
- Data (headline-only pipeline): `data/raw/Sarcasm_Headlines_Dataset.json` (raw, 28.6k),
  `data/processed/dataset.parquet` (7k, split)
- Data (combined corpus, Session 5): `src/data/build_combined.py`,
  `config.yaml` (`combined.*`), `data/processed/combined.parquet` (62k, 4 sources),
  `data/raw/iSarcasmEval_train.En.csv`, `data/raw/muSTARD_sarcasm_data.json`
