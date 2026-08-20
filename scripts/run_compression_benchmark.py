"""Phase 8 (stretch goal): software-only quantization and pruning of the
CNNGroupNorm baseline, CPU-only, reusing the Phase 6 latency-benchmarking
methodology (10 warmup + 100 timed calls, mean/p95).

Everything below is evaluated on a single held-out fold (machine_id=id_00,
same split/hyperparameters/seed as the Phase 4/5 CV run) rather than the full
4-fold CV, since re-running every compression variant across all 4 folds is
out of scope for a stretch goal -- this gives one consistent comparison point
across fp32 / int8 / pruned-30/50/70% variants.
"""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.compression import (
    benchmark_model_only_latency,
    evaluate_accuracy,
    model_size_kb,
    predict_proba,
    prune_global_unstructured,
    quantize_static,
    to_input_tensor,
    measured_sparsity,
    N_TIMED,
    N_WARMUP,
)
from src.cnn_model import CNNGroupNorm
from src.train_cnn import MANIFEST_PATH, RANDOM_STATE, SPECTROGRAMS_PATH, VAL_FRACTION, get_device, train_one_fold

MACHINE_ID_HELD_OUT = "id_00"
CHECKPOINT_PATH = os.path.join("data", "processed", "compression_models", "cnn_id00_holdout.pt")
RESULTS_OUT = os.path.join("data", "processed", "compression_results.csv")

CALIBRATION_SIZE = 200
PRUNE_LEVELS = [0.30, 0.50, 0.70]


def prepare_fold(machine_id: str = MACHINE_ID_HELD_OUT):
    """Identical split logic to the fold loop in src/train_cnn.py
    run_cnn_cross_validation, isolated to a single held-out machine."""
    manifest = pd.read_csv(MANIFEST_PATH)
    specs = np.load(SPECTROGRAMS_PATH)
    assert len(manifest) == len(specs), "manifest and spectrograms are misaligned"

    y_all = (manifest["label"] == "abnormal").astype(np.float32).values

    test_mask = (manifest["machine_id"] == machine_id).values
    train_val_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=VAL_FRACTION,
        stratify=y_all[train_val_idx],
        random_state=RANDOM_STATE,
    )

    X_train_raw, y_train = specs[train_idx], y_all[train_idx]
    X_val_raw, y_val = specs[val_idx], y_all[val_idx]
    X_test_raw, y_test = specs[test_idx], y_all[test_idx]

    mean, std = X_train_raw.mean(), X_train_raw.std()
    X_train = (X_train_raw - mean) / std
    X_val = (X_val_raw - mean) / std
    X_test = (X_test_raw - mean) / std
    return X_train, y_train, X_val, y_val, X_test, y_test, mean, std


def load_or_train_baseline(X_train, y_train, X_val, y_val) -> torch.nn.Module:
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Found existing id_00-holdout checkpoint at {CHECKPOINT_PATH}, loading (skipping retrain).")
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        model = CNNGroupNorm()
        model.load_state_dict(ckpt["state_dict"])
        return model.eval()

    print(f"No existing id_00-holdout checkpoint found; training one fold (held out={MACHINE_ID_HELD_OUT})...")
    device = get_device()
    model = train_one_fold(X_train, y_train, X_val, y_val, device, seed=RANDOM_STATE)
    model = model.to("cpu").eval()

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "machine_id_held_out": MACHINE_ID_HELD_OUT, "seed": RANDOM_STATE},
        CHECKPOINT_PATH,
    )
    print(f"Saved checkpoint to {CHECKPOINT_PATH}")
    return model


def main():
    X_train, y_train, X_val, y_val, X_test, y_test, mean, std = prepare_fold()
    print(f"Held-out machine: {MACHINE_ID_HELD_OUT}  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    model_fp32 = load_or_train_baseline(X_train, y_train, X_val, y_val)

    n_needed = N_WARMUP + N_TIMED
    assert len(X_test) >= n_needed, f"id_00 test set too small for latency benchmark ({len(X_test)} < {n_needed})"
    test_inputs = [to_input_tensor(X_test, i) for i in range(n_needed)]

    def fp32_forward(x):
        with torch.no_grad():
            model_fp32(x)

    rows = []

    # --- fp32 baseline ---
    print("\n=== fp32 baseline ===")
    proba = predict_proba(model_fp32, X_test)
    acc = evaluate_accuracy(proba, y_test)
    size_kb = model_size_kb(model_fp32.state_dict())
    lat = benchmark_model_only_latency(fp32_forward, test_inputs)
    print(f"size={size_kb:.1f} KB  latency mean/p95={lat['mean_ms']:.3f}/{lat['p95_ms']:.3f} ms  "
          f"f1={acc['f1']:.3f} roc_auc={acc['roc_auc']:.3f} pr_auc={acc['pr_auc']:.3f}")
    rows.append({"variant": "fp32_baseline", "size_kb": size_kb, "latency_mean_ms": lat["mean_ms"],
                 "latency_p95_ms": lat["p95_ms"], **acc, "measured_sparsity": 0.0, "quantization_fallback_layers": ""})

    # --- int8 static quantization ---
    print("\n=== int8 static quantization (calibration: training machines only) ===")
    rng = np.random.RandomState(RANDOM_STATE)
    calib_idx = rng.choice(len(X_train), size=min(CALIBRATION_SIZE, len(X_train)), replace=False)
    calibration_inputs = [to_input_tensor(X_train, i) for i in calib_idx]

    model_int8, fallback_types = quantize_static(model_fp32, calibration_inputs)
    if fallback_types:
        print(f"PARTIAL quantization: these layer types remain fp32 in the converted graph: {fallback_types} "
              f"(GroupNorm has no quantized CPU kernel in torch.ao.quantization -- Conv2d/Linear layers ARE "
              f"quantized, with dequantize/quantize boundaries inserted around each GroupNorm).")
    else:
        print("Full quantization: every layer was replaced with a quantized kernel.")

    def int8_forward(x):
        with torch.no_grad():
            model_int8(x)

    proba_int8 = predict_proba(model_int8, X_test)
    acc_int8 = evaluate_accuracy(proba_int8, y_test)
    size_int8 = model_size_kb(model_int8.state_dict())
    lat_int8 = benchmark_model_only_latency(int8_forward, test_inputs)
    print(f"size={size_int8:.1f} KB  latency mean/p95={lat_int8['mean_ms']:.3f}/{lat_int8['p95_ms']:.3f} ms  "
          f"f1={acc_int8['f1']:.3f} roc_auc={acc_int8['roc_auc']:.3f} pr_auc={acc_int8['pr_auc']:.3f}")
    rows.append({"variant": "int8_quantized", "size_kb": size_int8, "latency_mean_ms": lat_int8["mean_ms"],
                 "latency_p95_ms": lat_int8["p95_ms"], **acc_int8, "measured_sparsity": 0.0,
                 "quantization_fallback_layers": ",".join(fallback_types)})

    # --- unstructured magnitude pruning ---
    print("\n=== unstructured magnitude pruning (accuracy-vs-sparsity only, NOT a latency claim) ===")
    for amount in PRUNE_LEVELS:
        model_pruned = prune_global_unstructured(model_fp32, amount)
        sparsity = measured_sparsity(model_pruned)

        def pruned_forward(x, _m=model_pruned):
            with torch.no_grad():
                _m(x)

        proba_p = predict_proba(model_pruned, X_test)
        acc_p = evaluate_accuracy(proba_p, y_test)
        size_p = model_size_kb(model_pruned.state_dict())
        lat_p = benchmark_model_only_latency(pruned_forward, test_inputs)
        print(f"target={amount:.0%} measured={sparsity:.1%}  size={size_p:.1f} KB  "
              f"latency mean/p95={lat_p['mean_ms']:.3f}/{lat_p['p95_ms']:.3f} ms  "
              f"f1={acc_p['f1']:.3f} roc_auc={acc_p['roc_auc']:.3f} pr_auc={acc_p['pr_auc']:.3f}")
        rows.append({"variant": f"pruned_{int(amount*100)}pct", "size_kb": size_p,
                     "latency_mean_ms": lat_p["mean_ms"], "latency_p95_ms": lat_p["p95_ms"], **acc_p,
                     "measured_sparsity": sparsity, "quantization_fallback_layers": ""})

    results_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(RESULTS_OUT), exist_ok=True)
    results_df.to_csv(RESULTS_OUT, index=False)

    print("\n" + "=" * 100)
    print(f"Phase 8 compression summary (held out={MACHINE_ID_HELD_OUT}, CPU, n=100 timed calls after 10 warm-up)")
    print("=" * 100)
    print(results_df.to_string(index=False))
    print(
        "\nNote: pruning here demonstrates accuracy headroom, not a deployment speedup -- unstructured "
        "pruning zeroes weights without changing dense tensor shapes, so latency/size are expected to stay "
        "~flat across sparsity levels above. A real speedup would require structured pruning or a "
        "sparsity-aware runtime, out of scope here."
    )
    print(f"\nSaved to {os.path.abspath(RESULTS_OUT)}")


if __name__ == "__main__":
    main()
