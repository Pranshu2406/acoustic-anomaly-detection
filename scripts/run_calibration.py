"""CLI entrypoint for src/calibration.py's calibration pipeline.

Kept as a thin wrapper script (rather than a `if __name__ == "__main__"`
block inside src/calibration.py) deliberately: that module defines
`CalibratedPredictor`, which gets pickled via joblib. If src/calibration.py
were ever run directly (`python -m src.calibration`), Python would set that
module's `__name__` to `"__main__"`, and the pickled class would carry
`__module__ == "__main__"` -- unpicklable from any other entry point (e.g.
pytest), since `__main__` means something different there. Running the
pipeline from this separate script avoids that trap: `src.calibration` is
always imported normally, so `CalibratedPredictor` always pickles with its
real module path.
"""

import os

from src.calibration import CALIBRATOR_PATH, RELIABILITY_DIAGRAM_PATH, run_calibration_pipeline


def main():
    metrics_df, winner_name, rationale = run_calibration_pipeline()
    print("\n" + "=" * 80)
    print("Calibration: Brier score / ECE, raw vs. Platt vs. isotonic (held-out split)")
    print("=" * 80)
    print(metrics_df.to_string(index=False))
    print(f"\nWinner: {winner_name}")
    print(rationale)
    print(f"\nReliability diagram saved to {os.path.abspath(RELIABILITY_DIAGRAM_PATH)}")
    print(f"Calibrator artifact saved to {os.path.abspath(CALIBRATOR_PATH)}")


if __name__ == "__main__":
    main()
