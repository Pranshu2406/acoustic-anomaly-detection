"""Check whether id_06's (and id_02's) spectrogram input distribution differs
from the pooled training machines in each leave-one-machine-out fold --
direct evidence for/against the distribution-shift hypothesis before touching
the model architecture.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.train_cnn import MANIFEST_PATH, RANDOM_STATE, SPECTROGRAMS_PATH, VAL_FRACTION


def summarize(label, arr):
    print(
        f"  {label:42s} mean={arr.mean():8.4f}  std={arr.std():7.4f}  "
        f"min={arr.min():8.4f}  max={arr.max():7.4f}  n={arr.shape[0]}"
    )


def diagnose_fold(machine_id, manifest, specs, y_all):
    print(f"\n=== held-out machine: {machine_id} ===")
    test_mask = (manifest["machine_id"] == machine_id).values
    train_val_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]

    # Same inner split as src/train_cnn.py: z-scoring is fit on this train
    # portion only, exactly what's actually fed to the network during training.
    train_idx, _val_idx = train_test_split(
        train_val_idx,
        test_size=VAL_FRACTION,
        stratify=y_all[train_val_idx],
        random_state=RANDOM_STATE,
    )

    X_train_raw = specs[train_idx]
    X_test_raw = specs[test_idx]

    print(" -- raw log-mel values (before normalization) --")
    summarize(f"pooled training clips (excl. {machine_id})", X_train_raw)
    summarize(f"{machine_id} clips", X_test_raw)

    mean, std = X_train_raw.mean(), X_train_raw.std()
    X_train = (X_train_raw - mean) / std
    X_test = (X_test_raw - mean) / std

    print(f" -- z-scored, fit on training fold only (mean={mean:.4f}, std={std:.4f}) --")
    summarize(f"pooled training clips (excl. {machine_id})", X_train)
    summarize(f"{machine_id} clips", X_test)


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    specs = np.load(SPECTROGRAMS_PATH)
    y_all = (manifest["label"] == "abnormal").astype(np.float32).values

    diagnose_fold("id_06", manifest, specs, y_all)
    diagnose_fold("id_02", manifest, specs, y_all)


if __name__ == "__main__":
    main()
