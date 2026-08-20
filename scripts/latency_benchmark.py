"""Latency and efficiency benchmark across all 4 models, measured on CPU for
an apples-to-apples comparison reflecting resource-constrained edge deployment
-- even though CNN *training* may use MPS acceleration, inference latency here
is forced to CPU regardless of what's available, since that's the realistic
deployment scenario.

Models are fit once on the full dataset for this benchmark (not the Phase
3-5 leave-one-machine-out folds) -- this phase measures speed/size, not
accuracy, which is already covered elsewhere.
"""

import torch  # import before xgboost -- see tests/conftest.py for why

import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.baselines import FEATURE_COLS
from src.cnn_model import CNNGroupNorm
from src.features import extract_scalar_features
from src.preprocessing import wav_to_logmel
from src.train_cnn import MANIFEST_PATH, RANDOM_STATE, SPECTROGRAMS_PATH, VAL_FRACTION, get_device, train_one_fold

SCALAR_FEATURES_PATH = "data/processed/scalar_features.csv"
MODELS_DIR = os.path.join("data", "processed", "latency_models")
RESULTS_OUT = os.path.join("data", "processed", "latency_results.csv")

N_WARMUP = 10
N_TIMED = 100
SEED = RANDOM_STATE


def _latency_stats(latencies_ms: np.ndarray) -> dict:
    return {
        "mean_ms": float(np.mean(latencies_ms)),
        "std_ms": float(np.std(latencies_ms)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
    }


def _time_calls(fn, items) -> np.ndarray:
    """Call fn(item) once per item; first N_WARMUP discarded, next N_TIMED timed."""
    for item in items[:N_WARMUP]:
        fn(item)
    latencies = []
    for item in items[N_WARMUP : N_WARMUP + N_TIMED]:
        start = time.perf_counter()
        fn(item)
        latencies.append((time.perf_counter() - start) * 1000.0)
    return np.array(latencies)


def train_classical(scalar_df: pd.DataFrame):
    X = scalar_df[FEATURE_COLS].values
    y = (scalar_df["label"] == "abnormal").astype(int).values

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(X_scaled, y)

    xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric="logloss", random_state=SEED)
    xgb.fit(X, y)

    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=SEED)
    rf.fit(X, y)

    return scaler, lr, xgb, rf


def train_cnn(manifest: pd.DataFrame, specs: np.ndarray):
    y_all = (manifest["label"] == "abnormal").astype(np.float32).values
    idx = np.arange(len(manifest))
    train_idx, val_idx = train_test_split(idx, test_size=VAL_FRACTION, stratify=y_all, random_state=SEED)

    X_train_raw, y_train = specs[train_idx], y_all[train_idx]
    X_val_raw, y_val = specs[val_idx], y_all[val_idx]

    mean, std = X_train_raw.mean(), X_train_raw.std()
    X_train = (X_train_raw - mean) / std
    X_val = (X_val_raw - mean) / std

    device = get_device()
    print(f"Training CNN on {device} (benchmark latency is still measured on CPU regardless)", flush=True)
    model = train_one_fold(X_train, y_train, X_val, y_val, device, seed=SEED)
    model = model.to("cpu").eval()
    return model, float(mean), float(std)


def model_size_bytes(obj, path: str, is_torch_state_dict: bool = False) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if is_torch_state_dict:
        torch.save(obj, path)
    else:
        joblib.dump(obj, path)
    return os.path.getsize(path)


def main():
    scalar_df = pd.read_csv(SCALAR_FEATURES_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)
    specs = np.load(SPECTROGRAMS_PATH)
    assert len(manifest) == len(specs)

    print("Training classical models on full dataset...")
    scaler, lr, xgb, rf = train_classical(scalar_df)

    print("Training CNN...")
    cnn_model, cnn_mean, cnn_std = train_cnn(manifest, specs)

    # --- model sizes ---
    os.makedirs(MODELS_DIR, exist_ok=True)
    lr_size = model_size_bytes(lr, os.path.join(MODELS_DIR, "lr.joblib"))
    xgb_size = model_size_bytes(xgb, os.path.join(MODELS_DIR, "xgb.joblib"))
    rf_size = model_size_bytes(rf, os.path.join(MODELS_DIR, "rf.joblib"))
    cnn_size = model_size_bytes(cnn_model.state_dict(), os.path.join(MODELS_DIR, "cnn_state_dict.pt"), is_torch_state_dict=True)
    cnn_params = sum(p.numel() for p in cnn_model.parameters())

    # --- sample clips for timing (same 110 clips for every model, for a fair comparison) ---
    rng = np.random.RandomState(SEED)
    sample_filepaths = rng.choice(manifest["filepath"].values, size=N_WARMUP + N_TIMED, replace=False).tolist()

    scalar_by_filepath = scalar_df.set_index("filepath")
    filepath_to_spec_idx = {fp: i for i, fp in enumerate(manifest["filepath"].values)}

    # Pre-extracted (not timed) inputs for "model-only" latency.
    X_rows = scalar_by_filepath.loc[sample_filepaths, FEATURE_COLS].values
    X_rows_scaled = scaler.transform(X_rows)
    spec_rows_normalized = np.stack(
        [(specs[filepath_to_spec_idx[fp]] - cnn_mean) / cnn_std for fp in sample_filepaths]
    ).astype(np.float32)

    # --- model-only latency: input already in memory, isolates the model itself ---
    lr_model_only = _time_calls(lambda i: lr.predict_proba(X_rows_scaled[i : i + 1]), list(range(len(sample_filepaths))))
    xgb_model_only = _time_calls(lambda i: xgb.predict_proba(X_rows[i : i + 1]), list(range(len(sample_filepaths))))
    rf_model_only = _time_calls(lambda i: rf.predict_proba(X_rows[i : i + 1]), list(range(len(sample_filepaths))))

    @torch.no_grad()
    def _cnn_forward(i):
        x = torch.from_numpy(spec_rows_normalized[i]).unsqueeze(0).unsqueeze(0)
        cnn_model(x)

    cnn_model_only = _time_calls(_cnn_forward, list(range(len(sample_filepaths))))

    # --- full pipeline latency: raw filepath -> prediction, feature extraction included ---
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

    @torch.no_grad()
    def cnn_full(fp):
        logmel = wav_to_logmel(fp)
        normalized = ((logmel - cnn_mean) / cnn_std).astype(np.float32)
        x = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
        cnn_model(x)

    lr_full_lat = _time_calls(lr_full, sample_filepaths)
    xgb_full_lat = _time_calls(xgb_full, sample_filepaths)
    rf_full_lat = _time_calls(rf_full, sample_filepaths)
    cnn_full_lat = _time_calls(cnn_full, sample_filepaths)

    # --- assemble results ---
    rows = []
    for name, model_only, full_pipeline, size_desc in [
        ("logistic_regression", lr_model_only, lr_full_lat, f"{lr_size / 1024:.1f} KB"),
        ("xgboost", xgb_model_only, xgb_full_lat, f"{xgb_size / 1024:.1f} KB"),
        ("random_forest", rf_model_only, rf_full_lat, f"{rf_size / 1024:.1f} KB"),
        ("cnn", cnn_model_only, cnn_full_lat, f"{cnn_params:,} params / {cnn_size / 1024:.1f} KB"),
    ]:
        mo = _latency_stats(model_only)
        fp_stats = _latency_stats(full_pipeline)
        rows.append(
            {
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
        )

    results_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(RESULTS_OUT), exist_ok=True)
    results_df.to_csv(RESULTS_OUT, index=False)

    print("\n" + "=" * 100)
    print("Latency & efficiency summary (CPU, n=100 timed calls after 10 warm-up)")
    print("=" * 100)
    print(results_df.to_string(index=False))
    print(f"\nSaved to {os.path.abspath(RESULTS_OUT)}")


if __name__ == "__main__":
    main()
