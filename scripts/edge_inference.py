#!/usr/bin/env python3
"""Minimal standalone edge-inference script -- this is the script that would
actually run ON the device (Raspberry Pi / Jetson / similar), not a dev
machine. No training code, no dev-only dependencies (no pandas, no
scikit-learn, no matplotlib, no jupyter).

Runtime dependencies: numpy, librosa+soundfile (log-mel extraction -- must
exactly match training preprocessing, see src/preprocessing.py), and
onnxruntime. ONNX Runtime is used instead of full PyTorch specifically for
its much smaller memory footprint -- relevant down to something as
constrained as a Raspberry Pi Zero 2 W (512MB RAM total). See README "Edge
deployment readiness" for why ONNX Runtime is the recommended path on that
board and TorchScript/full PyTorch is not.

Usage:
    python scripts/edge_inference.py path/to/clip.wav
    python scripts/edge_inference.py path/to/clip.wav --model models/export/cnn_deployment.onnx
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort

# Import only the specific preprocessing function needed (not the whole
# src package's heavier modules) -- this still guarantees the exact same
# log-mel extraction as training, just without pulling in pandas/torch/etc.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import wav_to_logmel  # noqa: E402

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "export", "cnn_deployment.onnx")
DEFAULT_NORM_STATS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "export", "norm_stats.json")

ANOMALY_THRESHOLD = 0.5


def load_norm_stats(path: str = DEFAULT_NORM_STATS_PATH) -> tuple:
    with open(path) as f:
        stats = json.load(f)
    return stats["mean"], stats["std"]


def load_session(model_path: str = DEFAULT_MODEL_PATH) -> ort.InferenceSession:
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def predict(session: ort.InferenceSession, filepath: str, mean: float, std: float) -> dict:
    """Run one .wav file through the exact training preprocessing + ONNX
    model, timed end-to-end (preprocessing + model), matching the
    methodology used for every other latency number in this project."""
    start = time.perf_counter()

    logmel = wav_to_logmel(filepath)
    x = ((logmel - mean) / std).astype(np.float32)[np.newaxis, np.newaxis, :, :]

    input_name = session.get_inputs()[0].name
    logit = session.run(None, {input_name: x})[0].item()
    proba = 1.0 / (1.0 + np.exp(-logit))

    latency_ms = (time.perf_counter() - start) * 1000.0
    label = "ANOMALY DETECTED" if proba >= ANOMALY_THRESHOLD else "NORMAL"
    return {"filepath": filepath, "label": label, "proba": proba, "logit": logit, "latency_ms": latency_ms}


def main():
    parser = argparse.ArgumentParser(description="Standalone edge inference for the CNNGroupNorm anomaly detector.")
    parser.add_argument("wav_path", help="Path to a .wav file to classify.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to the exported ONNX model.")
    parser.add_argument("--norm-stats", default=DEFAULT_NORM_STATS_PATH, help="Path to norm_stats.json.")
    args = parser.parse_args()

    mean, std = load_norm_stats(args.norm_stats)
    session = load_session(args.model)

    result = predict(session, args.wav_path, mean, std)
    print(
        f"{result['label']}  proba={result['proba']:.4f}  logit={result['logit']:.4f}  "
        f"latency={result['latency_ms']:.1f}ms  file={result['filepath']}"
    )


if __name__ == "__main__":
    main()
