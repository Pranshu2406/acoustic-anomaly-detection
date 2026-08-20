"""Probability calibration for the deployed (all-4-machine) CNNGroupNorm model.

This answers a different question than the Phase 4/5 leave-one-machine-out
tests: not "does this model generalize to an unseen machine" but "are this
model's probabilities on new clips from KNOWN machines trustworthy enough to
use as probabilities" (e.g. for the uncertainty banding in src/streaming.py).
Both are legitimate evaluations; they must not be conflated -- see README.

Calibration is fit and evaluated on a held-out validation split of the
all-4-machine training run that the model never received a gradient update
from. The existing Phase 6 all-4-machine split (src/train_cnn.py via
scripts/latency_benchmark.py) stratifies only by label, not by machine_id, so
it isn't reused as-is.

**Session-leakage finding (see README):** a first version of this split used
a random stratified 85/15 draw (stratified by machine_id+label). Investigation
showed MIMII's sequential filenames encode real recording-session structure
(confirmed empirically via RMS-energy autocorrelation -- abnormal clips in
particular show sharp block/plateau transitions between distinct fault-
severity sessions), and the random split placed a training clip within 1
file index of ~98% of held-out clips -- i.e. nearly every "held-out" clip had
a near-duplicate neighbor in the training set. `train_or_load_deployment_model`
now uses a chronological split instead: for each (machine_id, label) group,
clips are ordered by filename index and the LAST 15% is held out, mirroring
the leave-one-machine-out discipline used elsewhere in this project by
keeping held-out clips temporally separate from training clips rather than
randomly interleaved. Split indices are persisted alongside the checkpoint so
calibration can prove, structurally, that it only ever loads the held-out rows.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from src.cnn_model import CNNGroupNorm
from src.train_cnn import (
    MANIFEST_PATH,
    RANDOM_STATE,
    SPECTROGRAMS_PATH,
    get_device,
    train_one_fold,
)

VAL_FRACTION = 0.15  # 85/15, per the task spec

DEPLOYMENT_CHECKPOINT_PATH = os.path.join("data", "processed", "deployment_model", "cnn_deployment.pt")
CALIBRATOR_PATH = os.path.join("data", "processed", "calibration", "calibrator.joblib")
RELIABILITY_DIAGRAM_PATH = os.path.join("notebooks", "calibration_reliability_diagram.png")

N_BINS = 10
# A calibrator only "wins" if it improves both metrics by at least this much;
# smaller differences are noise-level on a ~830-clip validation split and
# should not be reported as a real improvement.
MIN_MEANINGFUL_IMPROVEMENT = 0.005

BAND_LOW = 0.2
BAND_HIGH = 0.8


# --------------------------------------------------------------------------
# Deployment model: all 4 machines, chronological last-15% split
# --------------------------------------------------------------------------


def _file_index(filepath: str) -> int:
    """MIMII filenames are zero-padded sequential integers (e.g.
    00000142.wav) that encode recording/session order -- see README's
    session-leakage investigation."""
    return int(os.path.splitext(os.path.basename(filepath))[0])


def chronological_last_fraction_split(manifest: pd.DataFrame, val_fraction: float = VAL_FRACTION) -> tuple:
    """For each (machine_id, label) group, order clips by filename index and
    hold out the chronologically LAST `val_fraction` of that group, with the
    rest as train. Unlike a random draw, this keeps held-out clips temporally
    separate from training clips by a single boundary per group, instead of
    interleaved throughout -- avoiding the same-session leakage a random
    split produces (see module docstring / README).
    """
    file_num = manifest["filepath"].map(_file_index)
    train_idx, val_idx = [], []
    for _, group in manifest.assign(file_num=file_num).groupby(["machine_id", "label"], sort=False):
        group_sorted = group.sort_values("file_num")
        n = len(group_sorted)
        n_val = max(1, int(round(n * val_fraction)))
        train_idx.extend(group_sorted.index[: n - n_val].tolist())
        val_idx.extend(group_sorted.index[n - n_val :].tolist())
    return np.array(sorted(train_idx)), np.array(sorted(val_idx))


def train_or_load_deployment_model(checkpoint_path: str = DEPLOYMENT_CHECKPOINT_PATH):
    """Load the all-4-machine deployment checkpoint, training it if it
    doesn't exist yet. The checkpoint bundles the model weights, the input
    normalization stats (fit on the train rows only), and the exact
    train_idx/val_idx used -- so any later calibration step can re-derive
    the held-out split deterministically instead of re-splitting (which
    would risk silently drifting from what the model actually trained on).
    """
    if os.path.exists(checkpoint_path):
        # weights_only=False: this checkpoint bundles numpy index arrays
        # alongside the state_dict (not just tensors), and it's always a
        # locally self-generated file, never an untrusted download.
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    manifest = pd.read_csv(MANIFEST_PATH)
    specs = np.load(SPECTROGRAMS_PATH)
    assert len(manifest) == len(specs), "manifest and spectrograms are misaligned"

    y_all = (manifest["label"] == "abnormal").astype(np.float32).values

    train_idx, val_idx = chronological_last_fraction_split(manifest, VAL_FRACTION)

    X_train_raw, y_train = specs[train_idx], y_all[train_idx]
    X_val_raw, y_val = specs[val_idx], y_all[val_idx]

    # Normalization stats from the training rows only -- val_idx rows are
    # never touched here, matching the discipline used everywhere else in
    # this project (src/train_cnn.py, scripts/latency_benchmark.py).
    mean, std = float(X_train_raw.mean()), float(X_train_raw.std())
    X_train = (X_train_raw - mean) / std
    X_val = (X_val_raw - mean) / std

    device = get_device()
    print(
        f"Training deployment model on {device}: {len(train_idx)} train / {len(val_idx)} val "
        f"(chronological last-15% per machine_id+label group, all 4 machines)"
    )
    model = train_one_fold(X_train, y_train, X_val, y_val, device, seed=RANDOM_STATE)
    model = model.to("cpu").eval()

    checkpoint = {
        "state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "seed": RANDOM_STATE,
        "split_method": "chronological_last_fraction_per_machine_and_label",
    }
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved deployment checkpoint to {checkpoint_path}")
    return checkpoint


def load_deployment_model(checkpoint_path: str = DEPLOYMENT_CHECKPOINT_PATH):
    """Convenience wrapper: (model, mean, std) only, for callers (e.g.
    src/streaming.py) that don't need the split indices."""
    ckpt = train_or_load_deployment_model(checkpoint_path)
    model = CNNGroupNorm()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["mean"], ckpt["std"]


def load_held_out_calibration_split(checkpoint: dict):
    """Rebuild ONLY the held-out validation rows (val_idx) as normalized
    model inputs + labels, for calibration fitting/evaluation. Deliberately
    never loads or indexes checkpoint['train_idx'] -- the training rows
    never enter this function's scope, so calibration structurally cannot
    touch them.
    """
    manifest = pd.read_csv(MANIFEST_PATH)
    specs = np.load(SPECTROGRAMS_PATH)

    val_idx = checkpoint["val_idx"]
    mean, std = checkpoint["mean"], checkpoint["std"]

    y_val = (manifest["label"].values[val_idx] == "abnormal").astype(np.float32)
    X_val = (specs[val_idx] - mean) / std
    return X_val, y_val


# --------------------------------------------------------------------------
# Raw model scores
# --------------------------------------------------------------------------


@torch.no_grad()
def compute_raw_scores(model: torch.nn.Module, X: np.ndarray) -> tuple:
    """(raw_logits, raw_probas) for every row of X, one sample at a time."""
    model.eval()
    logits = np.empty(len(X), dtype=np.float64)
    for i in range(len(X)):
        x = torch.from_numpy(X[i]).unsqueeze(0).unsqueeze(0).float()
        logits[i] = model(x).item()
    probas = 1.0 / (1.0 + np.exp(-logits))
    return logits, probas


# --------------------------------------------------------------------------
# Calibrators
# --------------------------------------------------------------------------


def fit_platt_scaling(raw_logits: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Platt scaling: a 1D logistic regression on the model's raw logits."""
    lr = LogisticRegression()
    lr.fit(np.asarray(raw_logits).reshape(-1, 1), np.asarray(y))
    return lr


def platt_predict(lr: LogisticRegression, raw_logits) -> np.ndarray:
    raw_logits = np.atleast_1d(np.asarray(raw_logits, dtype=np.float64))
    return lr.predict_proba(raw_logits.reshape(-1, 1))[:, 1]


def fit_isotonic_regression(raw_probas: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(np.asarray(raw_probas), np.asarray(y))
    return iso


def isotonic_predict(iso: IsotonicRegression, raw_probas) -> np.ndarray:
    raw_probas = np.atleast_1d(np.asarray(raw_probas, dtype=np.float64))
    return iso.predict(raw_probas)


# --------------------------------------------------------------------------
# Calibration quality metrics
# --------------------------------------------------------------------------


def brier_score(y_true, proba) -> float:
    return float(brier_score_loss(np.asarray(y_true), np.asarray(proba)))


def reliability_curve(y_true, proba, n_bins: int = N_BINS):
    """Per-bin (mean predicted probability, observed frequency, count) over
    `n_bins` equal-width bins spanning [0, 1]. Empty bins are omitted."""
    y_true = np.asarray(y_true, dtype=np.float64)
    proba = np.asarray(proba, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    confidences, accuracies, counts = [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (proba >= lo) & (proba <= hi) if i == n_bins - 1 else (proba >= lo) & (proba < hi)
        if not mask.any():
            continue
        confidences.append(proba[mask].mean())
        accuracies.append(y_true[mask].mean())
        counts.append(int(mask.sum()))
    return np.array(confidences), np.array(accuracies), np.array(counts)


def expected_calibration_error(y_true, proba, n_bins: int = N_BINS) -> float:
    """ECE over `n_bins` equal-width bins: sum over bins of
    (bin_weight * |mean_predicted_proba - observed_frequency|)."""
    confidences, accuracies, counts = reliability_curve(y_true, proba, n_bins)
    n = len(np.asarray(y_true))
    if n == 0 or len(counts) == 0:
        return 0.0
    weights = counts / n
    return float(np.sum(weights * np.abs(confidences - accuracies)))


# --------------------------------------------------------------------------
# Reliability diagram
# --------------------------------------------------------------------------


def plot_reliability_diagram(y_val, curves: dict, n_bins: int = N_BINS, save_path: str = RELIABILITY_DIAGRAM_PATH) -> str:
    """`curves`: {label -> proba array}. Saves a reliability diagram (mean
    predicted probability vs. observed frequency per bin) with a
    perfect-calibration diagonal reference line.

    Points are plotted as unconnected markers sized by bin sample count,
    not joined by a line: this model's predictions are heavily concentrated
    near 0 and 1 (it separates the classes almost perfectly), so most of the
    handful of mid-confidence bins hold only a few samples each. Connecting
    those sparse, noisy bins with a line would visually imply a trend that
    isn't actually supported by the sample counts -- marker size makes that
    sparsity visible instead of hiding it.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration", zorder=1)

    for label, proba in curves.items():
        confidences, accuracies, counts = reliability_curve(y_val, proba, n_bins)
        if len(counts) == 0:
            continue
        sizes = 30.0 + 300.0 * (counts / counts.max())
        ax.scatter(confidences, accuracies, s=sizes, alpha=0.65, label=label, zorder=2)

    ax.set_xlabel("mean predicted probability (bin)")
    ax.set_ylabel("observed frequency (bin)")
    ax.set_title("Reliability diagram -- deployment model, held-out calibration split\n(marker size = samples in that bin)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", markerscale=0.5)
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# --------------------------------------------------------------------------
# End-to-end pipeline
# --------------------------------------------------------------------------


def _pick_winner(metrics: dict) -> tuple:
    """Pick whichever calibrator improves BOTH Brier and ECE over raw by at
    least MIN_MEANINGFUL_IMPROVEMENT; if both qualify, the larger ECE
    improvement wins (ECE is what banding thresholds actually depend on). If
    neither qualifies, returns ("none", ...) rather than picking one anyway.
    """
    raw = metrics["raw"]
    candidates = []
    for name in ("platt", "isotonic"):
        m = metrics[name]
        brier_gain = raw["brier"] - m["brier"]
        ece_gain = raw["ece"] - m["ece"]
        if brier_gain > MIN_MEANINGFUL_IMPROVEMENT and ece_gain > MIN_MEANINGFUL_IMPROVEMENT:
            candidates.append((name, ece_gain, brier_gain))

    if not candidates:
        return "none", (
            "Neither Platt scaling nor isotonic regression improved both Brier score and ECE by a "
            f"meaningful margin (> {MIN_MEANINGFUL_IMPROVEMENT}) over the raw probabilities on the "
            "held-out split -- reporting this plainly rather than picking a calibrator anyway. "
            "Uncalibrated (raw) probabilities are used for banding."
        )

    candidates.sort(key=lambda t: t[1], reverse=True)
    winner_name = candidates[0][0]
    return winner_name, (
        f"{winner_name} calibration improved both Brier score and ECE by a meaningful margin "
        f"(> {MIN_MEANINGFUL_IMPROVEMENT}) over raw probabilities on the held-out split."
    )


class CalibratedPredictor:
    """Wraps whichever calibrator won (or "none"), dispatching on raw logit
    (Platt) vs. raw probability (isotonic / identity) as needed, plus the
    three-way uncertainty banding used by src/streaming.py."""

    def __init__(self, method: str, platt=None, isotonic=None):
        assert method in ("platt", "isotonic", "none")
        self.method = method
        self.platt = platt
        self.isotonic = isotonic

    def calibrate(self, raw_logit: float, raw_proba: float) -> float:
        if self.method == "platt":
            return float(platt_predict(self.platt, raw_logit)[0])
        if self.method == "isotonic":
            return float(isotonic_predict(self.isotonic, raw_proba)[0])
        return float(raw_proba)

    @staticmethod
    def band(calibrated_proba: float, low: float = BAND_LOW, high: float = BAND_HIGH) -> str:
        if calibrated_proba < low:
            return "normal"
        if calibrated_proba > high:
            return "anomaly"
        return "uncertain"


def save_calibration_artifact(predictor: CalibratedPredictor, metrics: dict, path: str = CALIBRATOR_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"predictor": predictor, "metrics": metrics}, path)


def load_calibration_artifact(path: str = CALIBRATOR_PATH) -> CalibratedPredictor:
    return joblib.load(path)["predictor"]


def run_calibration_pipeline(
    checkpoint_path: str = DEPLOYMENT_CHECKPOINT_PATH,
    calibrator_out_path: str = CALIBRATOR_PATH,
    diagram_out_path: str = RELIABILITY_DIAGRAM_PATH,
) -> tuple:
    """Fit + evaluate both calibrators on the held-out split, pick a winner
    (or explicitly report that neither helps), save the reliability diagram
    and the calibrator artifact. Returns (metrics_df, winner, rationale).
    """
    checkpoint = train_or_load_deployment_model(checkpoint_path)
    model = CNNGroupNorm()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Only val_idx rows are ever loaded from here on -- see
    # load_held_out_calibration_split's docstring.
    X_val, y_val = load_held_out_calibration_split(checkpoint)
    raw_logits, raw_probas = compute_raw_scores(model, X_val)

    platt = fit_platt_scaling(raw_logits, y_val)
    isotonic = fit_isotonic_regression(raw_probas, y_val)

    platt_proba = platt_predict(platt, raw_logits)
    iso_proba = isotonic_predict(isotonic, raw_probas)

    metrics = {
        "raw": {"brier": brier_score(y_val, raw_probas), "ece": expected_calibration_error(y_val, raw_probas)},
        "platt": {"brier": brier_score(y_val, platt_proba), "ece": expected_calibration_error(y_val, platt_proba)},
        "isotonic": {"brier": brier_score(y_val, iso_proba), "ece": expected_calibration_error(y_val, iso_proba)},
    }
    winner_name, rationale = _pick_winner(metrics)

    predictor = CalibratedPredictor(
        method=winner_name,
        platt=platt if winner_name == "platt" else None,
        isotonic=isotonic if winner_name == "isotonic" else None,
    )
    save_calibration_artifact(predictor, metrics, calibrator_out_path)

    plot_reliability_diagram(
        y_val,
        {"raw (uncalibrated)": raw_probas, "platt": platt_proba, "isotonic": iso_proba},
        save_path=diagram_out_path,
    )

    metrics_df = pd.DataFrame(metrics).T.rename_axis("variant").reset_index()
    metrics_df["n_val"] = len(y_val)
    return metrics_df, winner_name, rationale
