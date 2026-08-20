"""Diagnose the below-chance CNN result on the id_06 leave-one-machine-out fold.

For id_06 specifically: (a) confusion matrix at the 0.5 threshold, (b) summary
stats / histogram of predicted probabilities on the held-out set, and (c) a
re-run with a different torch random seed (same data split, same
hyperparameters) to check whether the below-chance result reproduces or was a
one-off unlucky initialization/early-stopping point.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.train_cnn import (
    MANIFEST_PATH,
    RANDOM_STATE,
    SPECTROGRAMS_PATH,
    VAL_FRACTION,
    _predict_proba,
    get_device,
    train_one_fold,
)

MACHINE_ID = "id_06"
SEEDS = (RANDOM_STATE, RANDOM_STATE + 1000)


def run_fold(manifest, specs, y_all, device, seed):
    test_mask = (manifest["machine_id"] == MACHINE_ID).values
    train_val_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]

    # Data split held fixed across seeds -- only the model's own randomness
    # (weight init, dataloader shuffle order) changes between runs.
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

    model = train_one_fold(X_train, y_train, X_val, y_val, device, seed=seed)

    test_proba = _predict_proba(model, X_test, device)
    test_pred = (test_proba >= 0.5).astype(int)
    return y_test, test_proba, test_pred


def report(seed, y_test, test_proba, test_pred):
    chance = y_test.mean()
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])

    print(f"\n=== seed={seed} ===")
    print(
        f"f1={f1_score(y_test, test_pred):.3f}  roc_auc={roc_auc_score(y_test, test_proba):.3f}  "
        f"pr_auc={average_precision_score(y_test, test_proba):.3f}  chance_pr_auc={chance:.3f}"
    )

    print("confusion matrix (rows=true, cols=predicted; 0=normal, 1=abnormal):")
    print("           pred_normal  pred_abnormal")
    print(f"true_normal    {cm[0, 0]:6d}       {cm[0, 1]:6d}")
    print(f"true_abnormal  {cm[1, 0]:6d}       {cm[1, 1]:6d}")

    print(
        f"predicted probability stats: min={test_proba.min():.4f} max={test_proba.max():.4f} "
        f"mean={test_proba.mean():.4f} std={test_proba.std():.4f}"
    )
    hist, edges = np.histogram(test_proba, bins=10, range=(0.0, 1.0))
    print("histogram of predicted probabilities (10 bins over [0, 1]):")
    for count, lo, hi in zip(hist, edges[:-1], edges[1:]):
        print(f"  [{lo:.1f}, {hi:.1f}): {count:4d} {'#' * count}")

    # Split the probability stats by true class to see if predictions are
    # flipped (high prob on true negatives) vs. collapsed to a narrow range.
    proba_true_normal = test_proba[y_test == 0]
    proba_true_abnormal = test_proba[y_test == 1]
    print(
        f"proba | true_normal:   mean={proba_true_normal.mean():.4f} "
        f"std={proba_true_normal.std():.4f} (n={len(proba_true_normal)})"
    )
    print(
        f"proba | true_abnormal: mean={proba_true_abnormal.mean():.4f} "
        f"std={proba_true_abnormal.std():.4f} (n={len(proba_true_abnormal)})"
    )


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    specs = np.load(SPECTROGRAMS_PATH)
    y_all = (manifest["label"] == "abnormal").astype(np.float32).values

    device = get_device()
    print(f"Using device: {device}")

    for seed in SEEDS:
        y_test, test_proba, test_pred = run_fold(manifest, specs, y_all, device, seed)
        report(seed, y_test, test_proba, test_pred)


if __name__ == "__main__":
    main()
