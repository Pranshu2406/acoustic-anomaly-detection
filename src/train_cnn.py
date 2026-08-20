"""Train the CNN with the same 4-fold leave-one-machine-out structure as the
classical baselines (src/baselines.py), for a fair comparison.

Defaults to CNNGroupNorm (see src/cnn_model.py for why: BatchNorm collapsed
predictions to below-chance on 2 of 4 held-out machines under distribution
shift; GroupNorm resolved it). SmallCNN (BatchNorm) is still selectable via
`model_cls` for the ablation comparison in scripts/compare_cnn_norm.py.

For each fold, the 3 training machines' clips are split into an inner
train/val set (stratified random split -- val is only used for early
stopping, not final evaluation). The true generalization test is always the
held-out 4th machine, evaluated once training stops.
"""

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.baselines import run_cross_validation
from src.cnn_model import CNNGroupNorm

MANIFEST_PATH = "data/processed/manifest.csv"
SPECTROGRAMS_PATH = "data/processed/spectrograms.npy"
SCALAR_FEATURES_PATH = "data/processed/scalar_features.csv"

VAL_FRACTION = 0.15
BATCH_SIZE = 32
MAX_EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 1e-3
RANDOM_STATE = 42


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X_t = torch.from_numpy(X).unsqueeze(1)  # (N, 1, n_mels, n_frames)
    y_t = torch.from_numpy(y).float()
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def _predict_proba(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 128) -> np.ndarray:
    model.eval()
    loader = _make_loader(X, np.zeros(len(X), dtype=np.float32), batch_size, shuffle=False)
    probs = []
    for xb, _ in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    seed: int = RANDOM_STATE,
    model_cls=CNNGroupNorm,
) -> nn.Module:
    """Train with early stopping on validation PR-AUC; return the best-val model.

    `seed` controls torch's RNG (weight init + dataloader shuffle order) so a
    fold's training run is reproducible, and so it can be deliberately re-run
    with a different seed to check whether a given result is stable or was a
    one-off unlucky initialization/optimization trajectory.

    `model_cls` is the model class to instantiate. Defaults to CNNGroupNorm
    (the primary model); pass model_cls=SmallCNN to deliberately re-run the
    BatchNorm ablation instead.
    """
    torch.manual_seed(seed)
    train_loader = _make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)

    model = model_cls().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_pr_auc = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        val_proba = _predict_proba(model, X_val, device)
        val_pr_auc = average_precision_score(y_val, val_proba)

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"    epoch {epoch:2d}/{MAX_EPOCHS}  val_pr_auc={val_pr_auc:.3f}  "
            f"best={best_val_pr_auc:.3f}  patience={epochs_without_improvement}/{PATIENCE}",
            flush=True,
        )

        if epochs_without_improvement >= PATIENCE:
            print(f"    early stopping at epoch {epoch} (best val_pr_auc={best_val_pr_auc:.3f})", flush=True)
            break

    model.load_state_dict(best_state)
    return model


def run_cnn_cross_validation(model_cls=CNNGroupNorm, model_name: str = "cnn", return_predictions: bool = False):
    """Run 4-fold leave-one-machine-out CV for the CNN.

    If `return_predictions` is True, also returns a second dataframe with one
    row per (clip, model): filepath, machine_id, label, model, proba -- the
    held-out-fold predicted probability for every clip, for error analysis.
    """
    manifest = pd.read_csv(MANIFEST_PATH)
    specs = np.load(SPECTROGRAMS_PATH)
    assert len(manifest) == len(specs), "manifest and spectrograms are misaligned"

    y_all = (manifest["label"] == "abnormal").astype(np.float32).values
    machine_ids = sorted(manifest["machine_id"].unique())

    device = get_device()
    print(f"Using device: {device}  model: {model_name}", flush=True)

    results = []
    predictions = []
    for machine_id in machine_ids:
        fold_start = time.time()
        print(f"\n[fold start] model={model_name} held out = {machine_id}", flush=True)

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

        # Standardize with train-fold statistics only (same discipline as the
        # StandardScaler used for logistic regression in src/baselines.py).
        mean, std = X_train_raw.mean(), X_train_raw.std()
        X_train = (X_train_raw - mean) / std
        X_val = (X_val_raw - mean) / std
        X_test = (X_test_raw - mean) / std

        model = train_one_fold(X_train, y_train, X_val, y_val, device, model_cls=model_cls)

        test_proba = _predict_proba(model, X_test, device)
        test_pred = (test_proba >= 0.5).astype(int)
        chance_pr_auc = y_test.mean()

        fold_result = {
            "machine_id_held_out": machine_id,
            "model": model_name,
            "f1": f1_score(y_test, test_pred),
            "roc_auc": roc_auc_score(y_test, test_proba),
            "pr_auc": average_precision_score(y_test, test_proba),
            "chance_pr_auc": chance_pr_auc,
        }
        results.append(fold_result)

        if return_predictions:
            test_manifest = manifest.iloc[test_idx]
            predictions.append(
                pd.DataFrame(
                    {
                        "filepath": test_manifest["filepath"].values,
                        "machine_id": test_manifest["machine_id"].values,
                        "label": test_manifest["label"].values,
                        "model": model_name,
                        "proba": test_proba,
                    }
                )
            )

        elapsed = time.time() - fold_start
        print(
            f"[fold done] model={model_name} {machine_id}  f1={fold_result['f1']:.3f}  "
            f"roc_auc={fold_result['roc_auc']:.3f}  pr_auc={fold_result['pr_auc']:.3f}  "
            f"chance_pr_auc={chance_pr_auc:.3f}  ({elapsed:.1f}s)",
            flush=True,
        )

    results_df = pd.DataFrame(results)
    if return_predictions:
        return results_df, pd.concat(predictions, ignore_index=True)
    return results_df


def main():
    cnn_results = run_cnn_cross_validation()

    print("\n\nCNN per-fold results:")
    print(cnn_results.to_string(index=False))

    print("\nCNN summary (mean / std across 4 folds):")
    print(cnn_results.groupby("model")[["f1", "roc_auc", "pr_auc"]].agg(["mean", "std"]))

    # Compare against the best classical PR-AUC per machine (re-run for exact,
    # non-stale numbers rather than hardcoding Phase 3 results).
    scalar_df = pd.read_csv(SCALAR_FEATURES_PATH)
    classical_results = run_cross_validation(scalar_df)
    best_classical = classical_results.loc[
        classical_results.groupby("machine_id_held_out")["pr_auc"].idxmax()
    ][["machine_id_held_out", "model", "pr_auc"]].rename(
        columns={"model": "best_classical_model", "pr_auc": "best_classical_pr_auc"}
    )

    comparison = cnn_results.merge(best_classical, on="machine_id_held_out")
    comparison["cnn_beats_best_classical"] = comparison["pr_auc"] > comparison["best_classical_pr_auc"]

    print("\nCNN vs. best classical PR-AUC, per machine:")
    print(
        comparison[
            [
                "machine_id_held_out",
                "pr_auc",
                "best_classical_model",
                "best_classical_pr_auc",
                "cnn_beats_best_classical",
            ]
        ].to_string(index=False)
    )

    n_beats = comparison["cnn_beats_best_classical"].sum()
    print(f"\nCNN beats the best classical model on {n_beats}/{len(comparison)} individual machines.")


if __name__ == "__main__":
    main()
