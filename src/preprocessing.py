"""Raw MIMII .wav files -> fixed-size log-Mel spectrograms for CNN input."""

import glob
import os

import librosa
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64
N_FRAMES = 313  # 10s * 16000 Hz / 512 hop ~= 313, fixed width so every CNN input is the same shape

LABELS = ("normal", "abnormal")


def load_channel_0(path: str, expected_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load channel 0 of a MIMII .wav file.

    MIMII recordings are 8-channel (TAMAGO-03 microphone array), but we use
    channel 0 only and discard channels 1-7. This matches the MIMII baseline
    convention and most published work on this dataset, and keeps the initial
    model simple and interpretable. Multi-channel/array-based input is a
    documented possible follow-up experiment, not the default -- the same
    "don't add complexity without proving it's needed" reasoning behind the
    bank-failure literature's XGBoost-vs-logistic-regression result, where
    the more complex model wasn't adopted until it actually earned its keep.

    Raises ValueError instead of silently resampling if the file's sample
    rate doesn't match `expected_sr`, since a silent resample would quietly
    change the frequency content of every downstream feature.
    """
    waveform, file_sr = sf.read(path, always_2d=True)
    if file_sr != expected_sr:
        raise ValueError(
            f"{path}: expected {expected_sr} Hz, got {file_sr} Hz. "
            "Refusing to silently resample -- verify the source data."
        )
    return waveform[:, 0].astype(np.float32)


def pad_or_truncate(spec: np.ndarray, n_frames: int = N_FRAMES) -> np.ndarray:
    """Pad (with the spectrogram's own floor value) or truncate the time axis
    to exactly `n_frames` columns, so every output has identical shape.
    """
    current_frames = spec.shape[1]
    if current_frames == n_frames:
        return spec
    if current_frames > n_frames:
        return spec[:, :n_frames]
    pad_width = n_frames - current_frames
    floor = spec.min() if spec.size else 0.0
    return np.pad(spec, ((0, 0), (0, pad_width)), mode="constant", constant_values=floor)


def waveform_to_logmel(waveform: np.ndarray) -> np.ndarray:
    """Convert a raw 16kHz mono waveform to a fixed-size log-Mel spectrogram.

    Shared core of `wav_to_logmel` -- also called directly by real-time
    streaming inference (src/streaming.py), which has an in-memory audio
    buffer rather than a file on disk. Keeping this logic in one place means
    streaming inference can never silently drift from training preprocessing.

    Returns an (N_MELS, N_FRAMES) = (64, 313) array.
    """
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return pad_or_truncate(log_mel, N_FRAMES)


def wav_to_logmel(filepath: str) -> np.ndarray:
    """Convert a MIMII .wav file to a fixed-size log-Mel spectrogram.

    Returns an (N_MELS, N_FRAMES) = (64, 313) array.
    """
    waveform = load_channel_0(filepath)
    return waveform_to_logmel(waveform)


def build_manifest(raw_dir: str = "data/raw"):
    """Build a manifest of every .wav file under `raw_dir`.

    Columns: filepath, machine_type, machine_id, label, split.
    `split` is left as an empty placeholder -- train/test assignment happens
    later, grouped by machine_id so an entire machine unit (and every
    recording session that belongs to it) is held out together, preventing
    the same session from leaking across train and test.

    `pandas` is imported locally here rather than at module level, so that
    edge-deployment code (scripts/edge_inference.py) can import
    wav_to_logmel/load_channel_0 from this module without pulling in pandas
    -- a training/analysis-only dependency this function needs but nothing
    else in this module does.
    """
    import pandas as pd

    rows = []
    for filepath in sorted(glob.glob(os.path.join(raw_dir, "*", "*", "*", "*.wav"))):
        machine_type, machine_id, label = filepath.split(os.sep)[-4:-1]
        if label not in LABELS:
            continue
        rows.append(
            {
                "filepath": filepath,
                "machine_type": machine_type,
                "machine_id": machine_id,
                "label": label,
                "split": "",
            }
        )
    return pd.DataFrame(rows, columns=["filepath", "machine_type", "machine_id", "label", "split"])
