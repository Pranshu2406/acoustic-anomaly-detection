"""Audio -> Mel spectrogram preprocessing utilities."""

import numpy as np
import librosa


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """Load an audio file as a mono waveform resampled to `sr`."""
    waveform, _ = librosa.load(path, sr=sr, mono=True)
    return waveform


def to_mel_spectrogram(
    waveform: np.ndarray,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 64,
) -> np.ndarray:
    """Convert a waveform to a log-scaled Mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    return librosa.power_to_db(mel, ref=np.max)
