"""Phase 8: software-only compression (post-training static int8 quantization,
unstructured magnitude pruning) for CNNGroupNorm. CPU-only, no physical
hardware -- reuses the Phase 6 latency-benchmarking methodology (10 warmup +
100 timed calls, mean/p95) so compressed variants are directly comparable to
the Phase 6 fp32 numbers.

Quantization uses FX graph-mode PTQ (`torch.ao.quantization.quantize_fx`)
rather than eager-mode, because GroupNorm has no registered quantized CPU
kernel: FX mode automatically inserts dequantize/quantize boundaries around
unsupported layers instead of erroring, so the same call handles the
partially-quantizable architecture that this model is. The 'qnnpack' engine
is used because it's the only quantized CPU backend available on arm64
(Apple Silicon); 'fbgemm' is x86-only.
"""

import copy
import os
import tempfile
import time

import numpy as np
import torch
import torch.ao.quantization.quantize_fx as quantize_fx
import torch.nn.utils.prune as prune
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

QUANT_ENGINE = "qnnpack"
N_WARMUP = 10
N_TIMED = 100

# Module types in CNNGroupNorm that FX PTQ cannot replace with a quantized
# kernel and therefore leaves running in fp32 (with dequant/quant boundaries
# inserted around them).
KNOWN_UNQUANTIZABLE_TYPES = {"GroupNorm"}


def to_input_tensor(X: np.ndarray, i: int) -> torch.Tensor:
    """Single normalized spectrogram X[i] (shape (n_mels, n_frames)) -> model
    input tensor (1, 1, n_mels, n_frames)."""
    return torch.from_numpy(X[i]).unsqueeze(0).unsqueeze(0).float()


def model_size_kb(state_dict_or_model) -> float:
    """Serialized size via torch.save, matching the convention already used
    in scripts/latency_benchmark.py."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        torch.save(state_dict_or_model, path)
        return os.path.getsize(path) / 1024.0


def benchmark_model_only_latency(forward_fn, inputs: list) -> dict:
    """Same methodology as scripts/latency_benchmark.py: first N_WARMUP calls
    discarded, next N_TIMED calls timed individually. `inputs` must have at
    least N_WARMUP + N_TIMED elements."""
    assert len(inputs) >= N_WARMUP + N_TIMED, (
        f"need at least {N_WARMUP + N_TIMED} inputs, got {len(inputs)}"
    )
    for x in inputs[:N_WARMUP]:
        forward_fn(x)
    latencies = []
    for x in inputs[N_WARMUP : N_WARMUP + N_TIMED]:
        start = time.perf_counter()
        forward_fn(x)
        latencies.append((time.perf_counter() - start) * 1000.0)
    latencies = np.array(latencies)
    return {"mean_ms": float(latencies.mean()), "p95_ms": float(np.percentile(latencies, 95))}


def evaluate_accuracy(proba: np.ndarray, y_true: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


@torch.no_grad()
def predict_proba(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    """Sigmoid(model(x)) for every row of X, one sample at a time (matches how
    latency is measured -- no batching)."""
    model.eval()
    probs = np.empty(len(X), dtype=np.float32)
    for i in range(len(X)):
        logits = model(to_input_tensor(X, i))
        probs[i] = torch.sigmoid(logits).item()
    return probs


def quantize_static(model: torch.nn.Module, calibration_inputs: list) -> tuple:
    """Post-training static int8 quantization via FX graph mode.

    `calibration_inputs` must be drawn only from training-machine data (never
    the held-out test machine) to avoid calibration leakage.

    Returns (quantized_model, fallback_module_types) where fallback_module_types
    lists module class names that remain in fp32 in the converted graph (e.g.
    GroupNorm, which has no quantized CPU kernel) -- empty list means fully
    quantized, non-empty means partial quantization.
    """
    torch.backends.quantized.engine = QUANT_ENGINE
    model = copy.deepcopy(model).eval()
    example_input = calibration_inputs[0]

    qconfig_mapping = torch.ao.quantization.get_default_qconfig_mapping(QUANT_ENGINE)
    prepared = quantize_fx.prepare_fx(model, qconfig_mapping, example_input)

    with torch.no_grad():
        for x in calibration_inputs:
            prepared(x)

    converted = quantize_fx.convert_fx(prepared)

    fallback_types = sorted(
        {type(m).__name__ for m in converted.modules() if type(m).__name__ in KNOWN_UNQUANTIZABLE_TYPES}
    )
    return converted, fallback_types


def prune_global_unstructured(model: torch.nn.Module, amount: float) -> torch.nn.Module:
    """Unstructured global L1-magnitude pruning across every Conv2d/Linear
    weight tensor, at the given fraction (0-1) of weights set to exactly zero.

    Pruning masks are made permanent via `prune.remove` so the returned
    model's weight tensors are literally zeroed dense tensors -- this does
    NOT change tensor shapes/density, so it will not reduce latency or
    serialized size under standard dense CPU inference (see README Phase 8).
    """
    model = copy.deepcopy(model).eval()
    params_to_prune = [
        (m, "weight") for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))
    ]
    prune.global_unstructured(params_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
    for m, name in params_to_prune:
        prune.remove(m, name)
    return model


def measured_sparsity(model: torch.nn.Module) -> float:
    """Fraction of exactly-zero weights across all Conv2d/Linear weight tensors."""
    total = 0
    zeros = 0
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            w = m.weight.detach()
            total += w.numel()
            zeros += int((w == 0).sum().item())
    return zeros / total
