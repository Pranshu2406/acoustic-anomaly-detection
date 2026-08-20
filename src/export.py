"""Export the all-4-machine deployment CNNGroupNorm model
(src.calibration.load_deployment_model) to TorchScript and ONNX for edge
deployment -- no PyTorch training stack (or PyTorch at all, in the ONNX
Runtime case) needed on the target device.

Both `torch.jit.script` and `torch.jit.trace` work cleanly on this
architecture (verified empirically -- no data-dependent control flow to trip
up tracing, and every op GroupNorm included is TorchScript-scriptable).
`script` is preferred over `trace`: it doesn't bake in the specific input
shape/batch-size used at export time the way a traced graph does.

Normalization stats (mean/std) are exported alongside the model as a small
JSON file rather than requiring the edge script to load the full training
checkpoint just to read two floats.
"""

import json
import os

import torch

from src.calibration import load_deployment_model
from src.preprocessing import N_FRAMES, N_MELS

EXPORT_DIR = os.path.join("models", "export")
TORCHSCRIPT_PATH = os.path.join(EXPORT_DIR, "cnn_deployment.torchscript.pt")
ONNX_PATH = os.path.join(EXPORT_DIR, "cnn_deployment.onnx")
NORM_STATS_PATH = os.path.join(EXPORT_DIR, "norm_stats.json")

ONNX_OPSET = 17
ONNX_INPUT_NAME = "spectrogram"
ONNX_OUTPUT_NAME = "logit"


def export_torchscript(model: torch.nn.Module, out_path: str = TORCHSCRIPT_PATH) -> str:
    model = model.eval()
    scripted = torch.jit.script(model)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    scripted.save(out_path)
    return out_path


def export_onnx(model: torch.nn.Module, out_path: str = ONNX_PATH, opset: int = ONNX_OPSET) -> str:
    model = model.eval()
    example = torch.randn(1, 1, N_MELS, N_FRAMES)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.onnx.export(
        model,
        example,
        out_path,
        input_names=[ONNX_INPUT_NAME],
        output_names=[ONNX_OUTPUT_NAME],
        dynamic_axes={ONNX_INPUT_NAME: {0: "batch"}, ONNX_OUTPUT_NAME: {0: "batch"}},
        opset_version=opset,
    )
    return out_path


def export_norm_stats(mean: float, std: float, out_path: str = NORM_STATS_PATH) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"mean": mean, "std": std}, f, indent=2)
    return out_path


def export_all() -> dict:
    """Load the deployment model and export TorchScript + ONNX + norm stats.
    Returns the three output paths."""
    model, mean, std = load_deployment_model()
    ts_path = export_torchscript(model)
    onnx_path = export_onnx(model)
    stats_path = export_norm_stats(mean, std)
    return {"torchscript": ts_path, "onnx": onnx_path, "norm_stats": stats_path}


def load_torchscript(path: str = TORCHSCRIPT_PATH) -> torch.jit.ScriptModule:
    return torch.jit.load(path)


def load_norm_stats(path: str = NORM_STATS_PATH) -> tuple:
    with open(path) as f:
        stats = json.load(f)
    return stats["mean"], stats["std"]
