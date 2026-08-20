import os

import numpy as np
import pytest

from src.calibration import DEPLOYMENT_CHECKPOINT_PATH
from src.streaming import RollingAudioBuffer, stream_test_file

NORMAL_FILE = "data/raw/fan/id_00/normal/00000000.wav"
ABNORMAL_FILE = "data/raw/fan/id_00/abnormal/00000000.wav"

requires_dataset = pytest.mark.skipif(
    not (os.path.exists(NORMAL_FILE) and os.path.exists(DEPLOYMENT_CHECKPOINT_PATH)),
    reason="requires the local MIMII dataset and a trained deployment checkpoint (data/processed/deployment_model)",
)


def test_rolling_buffer_starts_as_fixed_size_silence():
    buf = RollingAudioBuffer(sample_rate=10, window_seconds=2.0)  # window_samples=20, tiny for a fast test
    assert buf.window_samples == 20
    window = buf.read()
    assert window.shape == (20,)
    assert np.all(window == 0.0)
    assert not buf.is_full


def test_rolling_buffer_slides_with_new_audio():
    buf = RollingAudioBuffer(sample_rate=10, window_seconds=2.0)  # window_samples=20

    buf.write(np.arange(1, 11, dtype=np.float32))  # [1..10], window still not full (10 real + 10 zero-pad)
    window = buf.read()
    assert window.shape == (20,)
    np.testing.assert_array_equal(window[:10], np.zeros(10, dtype=np.float32))
    np.testing.assert_array_equal(window[10:], np.arange(1, 11, dtype=np.float32))
    assert not buf.is_full

    buf.write(np.arange(11, 21, dtype=np.float32))  # [11..20], window now exactly full
    window = buf.read()
    np.testing.assert_array_equal(window, np.arange(1, 21, dtype=np.float32))
    assert buf.is_full

    buf.write(np.arange(21, 26, dtype=np.float32))  # 5 more samples -> oldest 5 fall off the front
    window = buf.read()
    np.testing.assert_array_equal(window, np.arange(6, 26, dtype=np.float32))
    assert buf.is_full


def test_rolling_buffer_write_larger_than_window_keeps_most_recent_tail():
    buf = RollingAudioBuffer(sample_rate=10, window_seconds=2.0)  # window_samples=20
    buf.write(np.arange(100, dtype=np.float32))  # a single chunk bigger than the whole window
    window = buf.read()
    np.testing.assert_array_equal(window, np.arange(80, 100, dtype=np.float32))
    assert buf.is_full


@requires_dataset
@pytest.mark.parametrize("filepath,expected_band", [(NORMAL_FILE, "normal"), (ABNORMAL_FILE, "anomaly")])
def test_stream_test_file_produces_correct_final_prediction(filepath, expected_band):
    # PIPELINE SANITY CHECK, not a generalization result: the loaded deployment
    # checkpoint is trained on all 4 machines (see src.calibration), so id_00
    # clips were in its own training data -- a correct prediction here only
    # confirms buffering/preprocessing/calibration/banding are wired together
    # correctly end-to-end, not accuracy on an unseen machine (that's the
    # Phase 4/5 leave-one-machine-out table).
    # interval_s == full clip length -> exactly one inference once the buffer
    # is fully loaded, plus the mandatory end-of-stream inference.
    results = stream_test_file(filepath, interval_s=10.0, chunk_seconds=0.1)

    assert len(results) >= 1
    final_proba, final_band = results[-1]
    assert isinstance(final_proba, float)
    assert 0.0 <= final_proba <= 1.0
    assert final_band in ("normal", "uncertain", "anomaly")

    assert final_band == expected_band
