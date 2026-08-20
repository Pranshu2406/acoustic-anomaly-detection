"""Real-time streaming inference for CNNGroupNorm.

Pipeline: microphone input (or a simulated .wav file, for testing without
hardware or a live anomalous sound) -> rolling 10-second audio buffer ->
src.preprocessing.waveform_to_logmel (the exact same preprocessing used in
training, not a reimplementation) -> CNNGroupNorm -> raw probability ->
calibrated probability (src.calibration) -> three-way uncertainty band ->
console alert.

Inference runs on a configurable interval (default every 2s) rather than on
every incoming audio chunk, so the pipeline doesn't peg the CPU polling a
model that only needs a fresh answer every couple of seconds.
"""

import argparse
import datetime
import threading
import time

import numpy as np
import torch

from src.calibration import (
    CalibratedPredictor,
    load_calibration_artifact,
    load_deployment_model,
    run_calibration_pipeline,
)
from src.preprocessing import SAMPLE_RATE, load_channel_0, waveform_to_logmel

WINDOW_SECONDS = 10.0  # matches training clip length
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)

DEFAULT_INFERENCE_INTERVAL_S = 2.0
DEFAULT_CHUNK_SECONDS = 0.1  # simulated mic callback block size in test mode

_BAND_DISPLAY = {
    "normal": "NORMAL",
    "uncertain": "UNCERTAIN - FLAG FOR REVIEW",
    "anomaly": "ANOMALY DETECTED",
}


class RollingAudioBuffer:
    """Fixed-size sliding window over the most recent WINDOW_SECONDS of
    mono audio at SAMPLE_RATE. Thread-safe: written from an audio callback
    thread, read from the inference loop.

    Before the window has been filled with real audio (e.g. right at
    startup), the unfilled portion is zero (silence) rather than garbage, so
    `read()` always returns a WINDOW_SAMPLES-length array.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, window_seconds: float = WINDOW_SECONDS):
        self.sample_rate = sample_rate
        self.window_samples = int(sample_rate * window_seconds)
        self._buffer = np.zeros(self.window_samples, dtype=np.float32)
        self._filled = 0
        self._lock = threading.Lock()

    def write(self, chunk: np.ndarray) -> None:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        n = len(chunk)
        with self._lock:
            if n >= self.window_samples:
                self._buffer[:] = chunk[-self.window_samples :]
                self._filled = self.window_samples
            elif n > 0:
                self._buffer = np.concatenate([self._buffer[n:], chunk])
                self._filled = min(self._filled + n, self.window_samples)

    def read(self) -> np.ndarray:
        with self._lock:
            return self._buffer.copy()

    @property
    def is_full(self) -> bool:
        with self._lock:
            return self._filled >= self.window_samples


def load_calibrated_predictor():
    """(model, mean, std, calibrator) for streaming inference. Trains/loads
    the all-4-machine deployment model (src.calibration.load_deployment_model)
    and the calibrator fit on its held-out split (building both, and the
    calibrator artifact, on first use if they don't exist yet)."""
    model, mean, std = load_deployment_model()
    try:
        calibrator = load_calibration_artifact()
    except FileNotFoundError:
        print("No calibrator artifact found; running calibration pipeline once to build one...")
        _, winner, rationale = run_calibration_pipeline()
        print(f"Calibration winner: {winner}. {rationale}")
        calibrator = load_calibration_artifact()
    return model, mean, std, calibrator


def predict_from_waveform(waveform: np.ndarray, model: torch.nn.Module, mean: float, std: float) -> tuple:
    """waveform (WINDOW_SAMPLES,) -> (raw_logit, raw_proba), using the exact
    same preprocessing (waveform_to_logmel) and normalization as training."""
    logmel = waveform_to_logmel(waveform)
    normalized = ((logmel - mean) / std).astype(np.float32)
    x = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logit = model(x).item()
    proba = 1.0 / (1.0 + np.exp(-logit))
    return logit, proba


def _print_alert(band: str, calibrated_proba: float, raw_proba: float, latency_ms: float) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    label = _BAND_DISPLAY[band]
    print(f"[{ts}] {label:26s} calibrated={calibrated_proba:.3f}  raw={raw_proba:.3f}  latency={latency_ms:.1f}ms")


def run_inference_cycle(
    buffer: RollingAudioBuffer, model: torch.nn.Module, mean: float, std: float, calibrator: CalibratedPredictor
) -> tuple:
    """One inference cycle, timed end-to-end (buffer read -> preprocessing ->
    model -> calibration -> banding), same timing approach as Phase 6
    (time.perf_counter around the full call). Prints the alert line and
    returns (calibrated_proba, band, latency_ms)."""
    start = time.perf_counter()
    waveform = buffer.read()
    raw_logit, raw_proba = predict_from_waveform(waveform, model, mean, std)
    calibrated_proba = calibrator.calibrate(raw_logit, raw_proba)
    band = calibrator.band(calibrated_proba)
    latency_ms = (time.perf_counter() - start) * 1000.0
    _print_alert(band, calibrated_proba, raw_proba, latency_ms)
    return calibrated_proba, band, latency_ms


def stream_microphone(interval_s: float = DEFAULT_INFERENCE_INTERVAL_S, duration_s: float = None) -> None:
    """Live microphone mode. Requires an actual input device; Ctrl+C to stop
    (or pass duration_s to stop automatically after N seconds)."""
    import sounddevice as sd

    model, mean, std, calibrator = load_calibrated_predictor()
    buffer = RollingAudioBuffer()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[stream warning] {status}")
        buffer.write(indata[:, 0])

    block_size = int(SAMPLE_RATE * DEFAULT_CHUNK_SECONDS)
    print(
        f"Starting microphone stream (sample_rate={SAMPLE_RATE} Hz, window={WINDOW_SECONDS}s, "
        f"inference every {interval_s}s, calibrator={calibrator.method}). Ctrl+C to stop."
    )
    start_time = time.time()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=block_size, callback=callback):
        try:
            while duration_s is None or (time.time() - start_time) < duration_s:
                time.sleep(interval_s)
                run_inference_cycle(buffer, model, mean, std, calibrator)
        except KeyboardInterrupt:
            print("\nStopped.")


def stream_test_file(
    filepath: str,
    interval_s: float = DEFAULT_INFERENCE_INTERVAL_S,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
) -> list:
    """Test/simulated mode: streams a real .wav file through the identical
    buffer + inference pipeline in small chunks, instead of a live
    microphone -- validates the whole pipeline without hardware or a live
    anomalous sound. Returns a list of (calibrated_proba, band) tuples, one
    per inference cycle that ran during the stream.
    """
    model, mean, std, calibrator = load_calibrated_predictor()
    buffer = RollingAudioBuffer()

    waveform = load_channel_0(filepath)
    chunk_samples = max(1, int(SAMPLE_RATE * chunk_seconds))
    chunks_per_inference = max(1, round(interval_s / chunk_seconds))

    print(
        f"Simulating live stream from {filepath} (chunk={chunk_seconds}s, inference every {interval_s}s, "
        f"calibrator={calibrator.method})"
    )

    results = []
    chunk_count = 0
    for start in range(0, len(waveform), chunk_samples):
        buffer.write(waveform[start : start + chunk_samples])
        chunk_count += 1
        if chunk_count % chunks_per_inference == 0:
            calibrated_proba, band, _ = run_inference_cycle(buffer, model, mean, std, calibrator)
            results.append((calibrated_proba, band))

    # Final inference on the fully-loaded window, in case the file length
    # didn't land exactly on an inference-interval boundary.
    calibrated_proba, band, _ = run_inference_cycle(buffer, model, mean, std, calibrator)
    results.append((calibrated_proba, band))
    return results


def main():
    parser = argparse.ArgumentParser(description="Real-time streaming inference for CNNGroupNorm anomaly detection.")
    parser.add_argument("--mode", choices=["mic", "test"], default="mic")
    parser.add_argument("--file", type=str, default=None, help="Path to a .wav file to stream in --mode test.")
    parser.add_argument("--interval", type=float, default=DEFAULT_INFERENCE_INTERVAL_S, help="Seconds between inference cycles.")
    parser.add_argument(
        "--duration", type=float, default=None, help="--mode mic only: stop after N seconds (default: run until Ctrl+C)."
    )
    args = parser.parse_args()

    if args.mode == "test":
        if not args.file:
            parser.error("--file is required in --mode test")
        stream_test_file(args.file, interval_s=args.interval)
    else:
        stream_microphone(interval_s=args.interval, duration_s=args.duration)


if __name__ == "__main__":
    main()
