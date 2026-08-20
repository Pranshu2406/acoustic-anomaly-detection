"""Hand-engineered scalar audio features for the classical ML baselines."""

import librosa
import numpy as np

from src.preprocessing import SAMPLE_RATE, load_channel_0


def extract_scalar_features(filepath: str) -> dict:
    """Compute per-clip scalar features (mean/std of several librosa descriptors).

    Returns a flat dict of 10 named scalar features: spectral centroid,
    spectral bandwidth, spectral rolloff, zero-crossing rate, and RMS energy,
    each summarized as mean and std across frames.
    """
    waveform = load_channel_0(filepath)

    # spectral_centroid/bandwidth/rolloff each recompute an STFT internally if
    # not given one -- with matching defaults (n_fft=2048, hop_length=512) a
    # single shared magnitude spectrogram produces bit-identical results (verified:
    # max abs diff 0.0) while computing the STFT once instead of three times.
    # zero_crossing_rate has no S= parameter (it's time-domain, no STFT to share).
    # rms(S=...) is a *different* algorithm from rms(y=...) (spectral-magnitude
    # vs. time-domain framing -- verified they diverge, max abs diff ~0.003), so
    # rms is deliberately left on y= to keep its value unchanged.
    S = np.abs(librosa.stft(waveform, n_fft=2048, hop_length=512))

    centroid = librosa.feature.spectral_centroid(S=S, sr=SAMPLE_RATE)
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=SAMPLE_RATE)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=SAMPLE_RATE)
    zcr = librosa.feature.zero_crossing_rate(y=waveform)
    rms = librosa.feature.rms(y=waveform)

    return {
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_centroid_std": float(np.std(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "spectral_bandwidth_std": float(np.std(bandwidth)),
        "spectral_rolloff_mean": float(np.mean(rolloff)),
        "spectral_rolloff_std": float(np.std(rolloff)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
    }
