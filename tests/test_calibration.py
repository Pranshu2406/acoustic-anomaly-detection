from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.calibration import (
    CalibratedPredictor,
    brier_score,
    chronological_last_fraction_split,
    expected_calibration_error,
    fit_isotonic_regression,
    fit_platt_scaling,
    isotonic_predict,
    platt_predict,
)


# --------------------------------------------------------------------------
# Chronological split: held-out clips must be the temporally-LAST fraction
# per (machine_id, label) group, never interleaved with training clips --
# see README's session-leakage investigation for why a random draw is unsafe
# here (MIMII filenames encode real recording-session order).
# --------------------------------------------------------------------------


def _synthetic_manifest(n_per_group: int = 40) -> pd.DataFrame:
    rows = []
    for machine_id in ["id_00", "id_02"]:
        for label in ["normal", "abnormal"]:
            for i in range(n_per_group):
                rows.append(
                    {
                        "filepath": f"data/raw/fan/{machine_id}/{label}/{i:08d}.wav",
                        "machine_id": machine_id,
                        "label": label,
                    }
                )
    return pd.DataFrame(rows)


def test_chronological_split_holds_out_last_fraction_per_group():
    manifest = _synthetic_manifest(n_per_group=40)
    train_idx, val_idx = chronological_last_fraction_split(manifest, val_fraction=0.15)

    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(train_idx) + len(val_idx) == len(manifest)

    for (machine_id, label), group in manifest.groupby(["machine_id", "label"]):
        group_val_nums = sorted(
            int(manifest.loc[i, "filepath"].split("/")[-1].split(".")[0]) for i in val_idx if i in group.index
        )
        group_train_nums = sorted(
            int(manifest.loc[i, "filepath"].split("/")[-1].split(".")[0]) for i in train_idx if i in group.index
        )
        # held-out file numbers are exactly the top ~15% (6 of 40), and every
        # one of them is numerically greater than every training file number
        # in the same group -- a single boundary, no interleaving.
        assert len(group_val_nums) == 6
        assert max(group_train_nums) < min(group_val_nums)


# --------------------------------------------------------------------------
# "Only fit on the held-out split, never training data" -- fit_platt_scaling
# and fit_isotonic_regression take exactly the arrays passed to them (no
# hidden reference to any global training set), so calibration is
# structurally incapable of touching training data as long as callers only
# ever pass held-out (val) arrays in. These tests prove that at the sklearn
# call boundary: they spy on LogisticRegression.fit / IsotonicRegression.fit
# and assert the data that actually reaches sklearn is exactly the held-out
# arrays passed in -- with no trace of a disjoint "training" array that was
# constructed alongside them but never passed to the calibrator.
# --------------------------------------------------------------------------


def test_fit_platt_scaling_only_touches_the_arrays_it_was_given():
    rng = np.random.RandomState(0)
    # A "training fold" that must never reach the calibrator.
    train_logits = rng.randn(500) + 10.0  # deliberately shifted, easy to detect if leaked
    train_y = np.zeros(500)

    # The actual held-out split.
    val_logits = rng.randn(60)
    val_y = (rng.rand(60) > 0.5).astype(np.float64)

    captured = {}
    original_fit = LogisticRegression.fit

    def spy_fit(self, X, y):
        captured["X"] = np.array(X)
        captured["y"] = np.array(y)
        return original_fit(self, X, y)

    with patch.object(LogisticRegression, "fit", spy_fit):
        fit_platt_scaling(val_logits, val_y)

    # Only the held-out split reached sklearn's fit call.
    assert captured["X"].shape[0] == len(val_logits)
    np.testing.assert_array_equal(captured["X"].ravel(), val_logits)
    np.testing.assert_array_equal(captured["y"], val_y)
    # None of the shifted training-fold values leaked in.
    assert not np.any(np.isin(captured["X"].ravel(), train_logits))


def test_fit_isotonic_regression_only_touches_the_arrays_it_was_given():
    rng = np.random.RandomState(1)
    train_probas = np.clip(rng.randn(500) * 0.01 + 0.99, 0.0, 1.0)  # deliberately near 1.0
    train_y = np.ones(500)

    val_probas = rng.rand(60)
    val_y = (rng.rand(60) > 0.5).astype(np.float64)

    captured = {}
    original_fit = IsotonicRegression.fit

    def spy_fit(self, X, y):
        captured["X"] = np.array(X)
        captured["y"] = np.array(y)
        return original_fit(self, X, y)

    with patch.object(IsotonicRegression, "fit", spy_fit):
        fit_isotonic_regression(val_probas, val_y)

    assert captured["X"].shape[0] == len(val_probas)
    np.testing.assert_array_equal(captured["X"].ravel(), val_probas)
    np.testing.assert_array_equal(captured["y"], val_y)
    assert not np.any(np.isin(captured["X"].ravel(), train_probas))


# --------------------------------------------------------------------------
# Brier score / ECE correctness on synthetic examples with known values.
# --------------------------------------------------------------------------


def test_brier_score_perfect_predictions_is_zero():
    y = np.array([0, 1, 0, 1])
    proba = np.array([0.0, 1.0, 0.0, 1.0])
    assert brier_score(y, proba) == pytest.approx(0.0)


def test_brier_score_known_value():
    # Brier = mean((p - y)^2)
    y = np.array([1, 0, 1, 0])
    proba = np.array([0.8, 0.2, 0.6, 0.4])
    expected = np.mean([(0.8 - 1) ** 2, (0.2 - 0) ** 2, (0.6 - 1) ** 2, (0.4 - 0) ** 2])
    assert brier_score(y, proba) == pytest.approx(expected)


def test_ece_perfectly_calibrated_bin_is_zero():
    # A single bin (all predictions ~0.9) where the observed frequency is
    # exactly 0.9 -- perfectly calibrated, ECE should be ~0.
    proba = np.array([0.9] * 10)
    y = np.array([1] * 9 + [0] * 1)  # observed frequency = 0.9
    ece = expected_calibration_error(y, proba, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_ece_known_miscalibration_value():
    # Two bins, evenly weighted, with known |confidence - accuracy| gaps:
    # bin A: 10 samples all predicted 0.9, but observed frequency 0.5 -> gap 0.4
    # bin B: 10 samples all predicted 0.1, but observed frequency 0.5 -> gap 0.4
    proba = np.array([0.9] * 10 + [0.1] * 10)
    y = np.array([1] * 5 + [0] * 5 + [1] * 5 + [0] * 5)
    ece = expected_calibration_error(y, proba, n_bins=10)
    # weight 0.5 each, gap 0.4 each -> ECE = 0.5*0.4 + 0.5*0.4 = 0.4
    assert ece == pytest.approx(0.4, abs=1e-9)


def test_calibration_improves_ece_on_synthetic_miscalibrated_scores():
    """A synthetic held-out split with KNOWN, deliberate miscalibration
    (raw probabilities systematically overconfident) -- confirms Platt
    scaling and isotonic regression both measurably reduce ECE/Brier
    relative to the raw (uncalibrated) scores, and that the improvement is
    computed correctly (not just asserted).
    """
    rng = np.random.RandomState(42)
    n = 2000
    true_p = rng.uniform(0.0, 1.0, n)
    y = (rng.uniform(0.0, 1.0, n) < true_p).astype(np.float64)

    # Deliberately overconfident raw probabilities: push everything toward
    # the extremes without changing the ranking (a classic case both Platt
    # and isotonic scaling should be able to correct).
    raw_proba = np.clip(0.5 + 1.8 * (true_p - 0.5), 1e-6, 1 - 1e-6)
    raw_logit = np.log(raw_proba / (1 - raw_proba))

    platt = fit_platt_scaling(raw_logit, y)
    iso = fit_isotonic_regression(raw_proba, y)
    platt_proba = platt_predict(platt, raw_logit)
    iso_proba = isotonic_predict(iso, raw_proba)

    raw_ece = expected_calibration_error(y, raw_proba)
    platt_ece = expected_calibration_error(y, platt_proba)
    iso_ece = expected_calibration_error(y, iso_proba)

    raw_brier = brier_score(y, raw_proba)
    platt_brier = brier_score(y, platt_proba)
    iso_brier = brier_score(y, iso_proba)

    assert platt_ece < raw_ece
    assert iso_ece < raw_ece
    assert platt_brier < raw_brier
    assert iso_brier < raw_brier


# --------------------------------------------------------------------------
# Three-way banding
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "proba,expected_band",
    [(0.0, "normal"), (0.19, "normal"), (0.2, "uncertain"), (0.5, "uncertain"), (0.8, "uncertain"), (0.81, "anomaly"), (1.0, "anomaly")],
)
def test_band_thresholds(proba, expected_band):
    assert CalibratedPredictor.band(proba) == expected_band


def test_calibrated_predictor_none_method_passes_through_raw_proba():
    predictor = CalibratedPredictor(method="none")
    assert predictor.calibrate(raw_logit=2.0, raw_proba=0.73) == pytest.approx(0.73)
