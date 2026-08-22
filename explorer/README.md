# Results Explorer

An interactive companion to the written project report
(`acoustic-anomaly-detection-report.pdf` at the project root), built to make the
results more explorable than a static PDF — filter by model, toggle metrics, and
drill into the per-machine breakdowns behind each headline number.

All data in `data/*.csv` is transcribed directly from the report's tables. Nothing
here is synthetic or re-derived — it's the same classical ML (Logistic Regression,
XGBoost, Random Forest) vs. PyTorch CNN comparison on the MIMII fan-machine dataset,
evaluated with 4-fold leave-one-machine-out cross-validation, including the
diagnosed BatchNorm failure, the GroupNorm fix, the corrected error-diversity
finding, and the corrected latency benchmark.

This folder is self-contained and purely additive: it does not modify or depend on
anything in `src/`, `models/`, `notebooks/`, `scripts/`, `tests/`, or the project's
`data/` folder.

## Structure

```
explorer/
├── app.py              # Streamlit UI (tabs: Overview, Classical ML, CNN Deep Dive,
│                        #   Head-to-Head, Error Analysis, Latency & Efficiency, Compression)
├── utils.py             # Data loading + Plotly chart-building helpers
├── requirements.txt      # streamlit, pandas, plotly
├── data/                 # Report tables as CSVs (see below)
└── README.md
```

## Data files

| File | Report section |
|---|---|
| `classical_ml_results.csv` | §4 — per-machine F1 / ROC-AUC / PR-AUC for LR, XGBoost, RF |
| `classical_ml_summary.csv` | §4 — mean ± std across folds, per model |
| `cnn_results.csv` | §5 — BatchNorm vs. GroupNorm PR-AUC per machine |
| `cnn_summary.csv` | §5.3 — mean ± std across folds, BatchNorm CNN vs. GroupNorm CNN |
| `cnn_vs_classical.csv` | §5.4 — final CNN vs. best-classical comparison |
| `error_analysis.csv` | §6.1–6.2 — FP/FN counts, unique-error-rate correction |
| `fp_rate_by_machine.csv` | §6.3 — false-positive rate by held-out machine |
| `latency_results.csv` | §7 — model size, model-only and full-pipeline latency |
| `latency_correction.csv` | §7 — the 3.4x → 2.4x benchmarking-artifact correction |
| `compression_results.csv` | §8 — quantization and pruning (single-fold, stretch goal) |

## Running locally

From the project root:

```bash
pip install -r explorer/requirements.txt
streamlit run explorer/app.py
```

The app opens at `http://localhost:8501`.

## Deploying on Streamlit Community Cloud

Point the deployment at this repo with:
- **Main file path:** `explorer/app.py`
- **Requirements file:** `explorer/requirements.txt`

No secrets or external services are required — the app only reads the CSVs in
`explorer/data/`.
