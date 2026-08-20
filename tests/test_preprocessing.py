import numpy as np
import pytest

from src.preprocessing import N_FRAMES, N_MELS, pad_or_truncate, wav_to_logmel

NORMAL_FILE = "data/raw/fan/id_00/normal/00000000.wav"
ABNORMAL_FILE = "data/raw/fan/id_00/abnormal/00000000.wav"


@pytest.mark.parametrize("filepath", [NORMAL_FILE, ABNORMAL_FILE])
def test_wav_to_logmel_shape_on_real_file(filepath):
    logmel = wav_to_logmel(filepath)
    assert logmel.shape == (N_MELS, N_FRAMES)


def test_pad_or_truncate_pads_short_array():
    short = np.random.randn(N_MELS, 100).astype(np.float32)
    padded = pad_or_truncate(short, N_FRAMES)
    assert padded.shape == (N_MELS, N_FRAMES)
    np.testing.assert_array_equal(padded[:, :100], short)
    assert np.all(padded[:, 100:] == short.min())


def test_pad_or_truncate_truncates_long_array():
    long = np.random.randn(N_MELS, 500).astype(np.float32)
    truncated = pad_or_truncate(long, N_FRAMES)
    assert truncated.shape == (N_MELS, N_FRAMES)
    np.testing.assert_array_equal(truncated, long[:, :N_FRAMES])
