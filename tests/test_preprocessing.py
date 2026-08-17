import numpy as np

from src.preprocessing import to_mel_spectrogram


def test_to_mel_spectrogram_shape():
    sr = 16000
    waveform = np.zeros(sr, dtype=np.float32)  # 1 second of silence
    mel = to_mel_spectrogram(waveform, sr=sr, n_mels=64)
    assert mel.shape[0] == 64
    assert mel.ndim == 2
