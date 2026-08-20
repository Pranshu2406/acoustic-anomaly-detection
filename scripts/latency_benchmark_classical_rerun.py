"""Re-run the Phase 6 latency benchmark for the classical models only, after
fixing src/features.py to share a single STFT across spectral_centroid/
bandwidth/rolloff instead of each recomputing it. CNN numbers are untouched
(its pipeline doesn't use extract_scalar_features) and are carried over from
the existing data/processed/latency_results.csv.

Reuses the exact same methodology (same seed, same warm-up/timed counts, same
sample clips) as scripts/latency_benchmark.py for a fair before/after comparison.
"""

import os

import numpy as np
import pandas as pd

from scripts.latency_benchmark import (
    MODELS_DIR,
    N_TIMED,
    N_WARMUP,
    RESULTS_OUT,
    SEED,
    SCALAR_FEATURES_PATH,
    _latency_stats,
    _time_calls,
    model_size_bytes,
    train_classical,
)
from src.baselines import FEATURE_COLS
from src.features import extract_scalar_features
from src.preprocessing import build_manifest


def main():
    scalar_df = pd.read_csv(SCALAR_FEATURES_PATH)
    manifest = build_manifest("data/raw")

    print("Training classical models on full dataset (same as before)...")
    scaler, lr, xgb, rf = train_classical(scalar_df)

    lr_size = model_size_bytes(lr, os.path.join(MODELS_DIR, "lr.joblib"))
    xgb_size = model_size_bytes(xgb, os.path.join(MODELS_DIR, "xgb.joblib"))
    rf_size = model_size_bytes(rf, os.path.join(MODELS_DIR, "rf.joblib"))

    rng = np.random.RandomState(SEED)
    sample_filepaths = rng.choice(manifest["filepath"].values, size=N_WARMUP + N_TIMED, replace=False).tolist()

    scalar_by_filepath = scalar_df.set_index("filepath")
    X_rows = scalar_by_filepath.loc[sample_filepaths, FEATURE_COLS].values
    X_rows_scaled = scaler.transform(X_rows)

    lr_model_only = _time_calls(lambda i: lr.predict_proba(X_rows_scaled[i : i + 1]), list(range(len(sample_filepaths))))
    xgb_model_only = _time_calls(lambda i: xgb.predict_proba(X_rows[i : i + 1]), list(range(len(sample_filepaths))))
    rf_model_only = _time_calls(lambda i: rf.predict_proba(X_rows[i : i + 1]), list(range(len(sample_filepaths))))

    def lr_full(fp):
        feats = extract_scalar_features(fp)
        x = np.array([[feats[c] for c in FEATURE_COLS]])
        lr.predict_proba(scaler.transform(x))

    def xgb_full(fp):
        feats = extract_scalar_features(fp)
        x = np.array([[feats[c] for c in FEATURE_COLS]])
        xgb.predict_proba(x)

    def rf_full(fp):
        feats = extract_scalar_features(fp)
        x = np.array([[feats[c] for c in FEATURE_COLS]])
        rf.predict_proba(x)

    lr_full_lat = _time_calls(lr_full, sample_filepaths)
    xgb_full_lat = _time_calls(xgb_full, sample_filepaths)
    rf_full_lat = _time_calls(rf_full, sample_filepaths)

    new_rows = {}
    for name, model_only, full_pipeline, size_desc in [
        ("logistic_regression", lr_model_only, lr_full_lat, f"{lr_size / 1024:.1f} KB"),
        ("xgboost", xgb_model_only, xgb_full_lat, f"{xgb_size / 1024:.1f} KB"),
        ("random_forest", rf_model_only, rf_full_lat, f"{rf_size / 1024:.1f} KB"),
    ]:
        mo = _latency_stats(model_only)
        fp_stats = _latency_stats(full_pipeline)
        new_rows[name] = {
            "model": name,
            "size_desc": size_desc,
            "model_only_mean_ms": mo["mean_ms"],
            "model_only_std_ms": mo["std_ms"],
            "model_only_p95_ms": mo["p95_ms"],
            "full_pipeline_mean_ms": fp_stats["mean_ms"],
            "full_pipeline_std_ms": fp_stats["std_ms"],
            "full_pipeline_p95_ms": fp_stats["p95_ms"],
            "throughput_clips_per_sec": 1000.0 / fp_stats["mean_ms"],
        }

    old_results = pd.read_csv(RESULTS_OUT)
    print("\nBefore (classical rows only):")
    print(old_results[old_results["model"] != "cnn"].to_string(index=False))

    updated = old_results.set_index("model")
    for name, row in new_rows.items():
        updated.loc[name] = row
    updated = updated.reset_index()[old_results.columns]

    updated.to_csv(RESULTS_OUT, index=False)

    print("\nAfter (all rows, CNN carried over unchanged):")
    print(updated.to_string(index=False))
    print(f"\nSaved to {os.path.abspath(RESULTS_OUT)}")


if __name__ == "__main__":
    main()
