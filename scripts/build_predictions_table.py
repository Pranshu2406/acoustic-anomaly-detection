"""Build a single wide table of every model's held-out-fold prediction for
every clip, for cross-model error analysis.

Import order note: torch must be imported before xgboost in this process, or
the first torch op segfaults (bundled OpenMP conflict on this platform -- see
tests/conftest.py for the same fix applied to the test suite).
"""

import torch  # noqa: F401  -- import before xgboost, see module docstring

import os

import pandas as pd

from src.baselines import run_cross_validation
from src.train_cnn import run_cnn_cross_validation

SCALAR_FEATURES_PATH = "data/processed/scalar_features.csv"
OUT_PATH = os.path.join("data", "processed", "all_predictions.csv")


def main():
    scalar_df = pd.read_csv(SCALAR_FEATURES_PATH)
    _classical_results, classical_predictions = run_cross_validation(scalar_df, return_predictions=True)

    _cnn_results, cnn_predictions = run_cnn_cross_validation(return_predictions=True)

    long_predictions = pd.concat([classical_predictions, cnn_predictions], ignore_index=True)

    wide = long_predictions.pivot(index=["filepath", "machine_id", "label"], columns="model", values="proba")
    wide = wide.reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={c: f"proba_{c}" for c in wide.columns if c not in ("filepath", "machine_id", "label")})

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    wide.to_csv(OUT_PATH, index=False)

    print(f"Rows: {len(wide)}")
    print(f"Columns: {list(wide.columns)}")
    print(f"Any missing predictions (should be none): {wide.isna().any().any()}")
    print(f"Saved to {os.path.abspath(OUT_PATH)}")


if __name__ == "__main__":
    main()
