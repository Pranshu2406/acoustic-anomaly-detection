import pytest
import torch

from src.cnn_model import CNNGroupNorm
from src.compression import measured_sparsity, prune_global_unstructured, quantize_static


def _calibration_inputs(n: int = 8):
    return [torch.randn(1, 1, 64, 313) for _ in range(n)]


def test_quantized_model_output_shape_single_sample():
    model = CNNGroupNorm().eval()
    quant_model, _ = quantize_static(model, _calibration_inputs())
    x = torch.randn(1, 1, 64, 313)
    with torch.no_grad():
        out = quant_model(x)
    assert out.shape == (1,)


def test_quantized_model_output_shape_batch():
    model = CNNGroupNorm().eval()
    quant_model, _ = quantize_static(model, _calibration_inputs())
    x = torch.randn(4, 1, 64, 313)
    with torch.no_grad():
        out = quant_model(x)
    assert out.shape == (4,)


def test_quantized_model_reports_groupnorm_fallback():
    # GroupNorm has no quantized CPU kernel, so quantize_static must report it
    # explicitly rather than silently producing a fully-quantized model.
    model = CNNGroupNorm().eval()
    _, fallback_types = quantize_static(model, _calibration_inputs())
    assert "GroupNorm" in fallback_types


@pytest.mark.parametrize("target_sparsity", [0.30, 0.50, 0.70])
def test_pruning_matches_target_sparsity(target_sparsity):
    model = CNNGroupNorm().eval()
    pruned = prune_global_unstructured(model, amount=target_sparsity)
    sparsity = measured_sparsity(pruned)
    assert abs(sparsity - target_sparsity) < 0.02


def test_pruned_model_output_shape():
    model = CNNGroupNorm().eval()
    pruned = prune_global_unstructured(model, amount=0.5)
    x = torch.randn(4, 1, 64, 313)
    with torch.no_grad():
        out = pruned(x)
    assert out.shape == (4,)
