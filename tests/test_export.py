import os

import numpy as np
import pytest
import torch

from src.calibration import load_deployment_model
from src.export import (
    ONNX_INPUT_NAME,
    export_all,
    load_norm_stats,
    load_torchscript,
)
from src.preprocessing import wav_to_logmel

pytest.importorskip("onnxruntime")
import onnxruntime as ort  # noqa: E402

REAL_CLIPS = [
    "data/raw/fan/id_00/normal/00000000.wav",
    "data/raw/fan/id_00/abnormal/00000000.wav",
    "data/raw/fan/id_02/normal/00000000.wav",
    "data/raw/fan/id_02/abnormal/00000000.wav",
    "data/raw/fan/id_06/abnormal/00000005.wav",
]

requires_dataset = pytest.mark.skipif(
    not all(os.path.exists(f) for f in REAL_CLIPS), reason="requires the local MIMII dataset"
)

TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def exported_paths():
    return export_all()


@pytest.fixture(scope="module")
def torch_model():
    model, mean, std = load_deployment_model()
    return model, mean, std


def _make_input(filepath: str, mean: float, std: float) -> torch.Tensor:
    logmel = wav_to_logmel(filepath)
    normalized = ((logmel - mean) / std).astype(np.float32)
    return torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)


@requires_dataset
@pytest.mark.parametrize("filepath", REAL_CLIPS)
def test_torchscript_matches_pytorch(exported_paths, torch_model, filepath):
    model, mean, std = torch_model
    x = _make_input(filepath, mean, std)

    with torch.no_grad():
        torch_out = model(x).numpy()

    scripted = load_torchscript(exported_paths["torchscript"])
    with torch.no_grad():
        ts_out = scripted(x).numpy()

    np.testing.assert_allclose(ts_out, torch_out, atol=TOLERANCE)


@requires_dataset
@pytest.mark.parametrize("filepath", REAL_CLIPS)
def test_onnx_matches_pytorch(exported_paths, torch_model, filepath):
    model, mean, std = torch_model
    x = _make_input(filepath, mean, std)

    with torch.no_grad():
        torch_out = model(x).numpy()

    sess = ort.InferenceSession(exported_paths["onnx"], providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {ONNX_INPUT_NAME: x.numpy()})[0]

    assert onnx_out.shape == torch_out.shape
    np.testing.assert_allclose(onnx_out, torch_out, atol=TOLERANCE)


@requires_dataset
def test_onnx_matches_pytorch_batched(exported_paths, torch_model):
    """Confirms the dynamic batch axis actually works, not just batch=1."""
    model, mean, std = torch_model
    xs = torch.cat([_make_input(f, mean, std) for f in REAL_CLIPS], dim=0)

    with torch.no_grad():
        torch_out = model(xs).numpy()

    sess = ort.InferenceSession(exported_paths["onnx"], providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {ONNX_INPUT_NAME: xs.numpy()})[0]

    assert onnx_out.shape == torch_out.shape

    np.testing.assert_allclose(onnx_out, torch_out, atol=TOLERANCE)


def test_norm_stats_round_trip(exported_paths, torch_model):
    _, mean, std = torch_model
    loaded_mean, loaded_std = load_norm_stats(exported_paths["norm_stats"])
    assert loaded_mean == pytest.approx(mean)
    assert loaded_std == pytest.approx(std)
