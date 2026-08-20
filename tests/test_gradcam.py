import numpy as np
import torch

from src.cnn_model import CNNGroupNorm
from src.gradcam import GradCAM, summarize_cam
from src.preprocessing import N_FRAMES, N_MELS


def test_cam_output_shape_matches_input_spectrogram():
    torch.manual_seed(0)
    model = CNNGroupNorm().eval()
    x = torch.randn(1, 1, N_MELS, N_FRAMES)

    with GradCAM(model) as cam_fn:
        result = cam_fn(x)

    assert result["cam"].shape == (N_MELS, N_FRAMES)


def test_cam_values_are_finite_not_nan():
    torch.manual_seed(1)
    model = CNNGroupNorm().eval()
    x = torch.randn(1, 1, N_MELS, N_FRAMES)

    with GradCAM(model) as cam_fn:
        result = cam_fn(x)

    assert np.isfinite(result["cam"]).all()


def test_cam_values_are_not_degenerate_constant():
    # A GroupNorm-based CNN has no BatchNorm running-stat assumptions baked
    # into Grad-CAM's global-average-pooled-gradient weighting, but that's
    # verified here empirically rather than just assumed: the CAM must
    # actually vary across space, not collapse to a single repeated value
    # (which would happen if, e.g., gradients vanished or hooks silently
    # failed to fire).
    torch.manual_seed(2)
    model = CNNGroupNorm().eval()
    x = torch.randn(1, 1, N_MELS, N_FRAMES)

    with GradCAM(model) as cam_fn:
        result = cam_fn(x)

    cam = result["cam"]
    assert cam.std() > 1e-6
    assert cam.min() < cam.max()


def test_cam_is_min_max_normalized_to_unit_range():
    torch.manual_seed(3)
    model = CNNGroupNorm().eval()
    x = torch.randn(1, 1, N_MELS, N_FRAMES)

    with GradCAM(model) as cam_fn:
        result = cam_fn(x)

    cam = result["cam"]
    assert cam.min() == 0.0
    assert cam.max() == 1.0
    assert cam.min() >= 0.0 and cam.max() <= 1.0


def test_cam_differs_for_different_inputs():
    # Sanity check that the CAM is actually input-dependent, not some
    # fixed/degenerate pattern the model always produces regardless of x.
    torch.manual_seed(4)
    model = CNNGroupNorm().eval()
    x1 = torch.randn(1, 1, N_MELS, N_FRAMES)
    x2 = torch.randn(1, 1, N_MELS, N_FRAMES)

    with GradCAM(model) as cam_fn:
        cam1 = cam_fn(x1)["cam"]
        cam2 = cam_fn(x2)["cam"]

    assert not np.allclose(cam1, cam2)


def test_positive_and_negative_target_sign_produce_different_cams():
    # target_sign=+1 explains "abnormal" evidence, -1 explains "normal"
    # evidence (there's only one sigmoid output, not two class logits).
    torch.manual_seed(5)
    model = CNNGroupNorm().eval()
    x = torch.randn(1, 1, N_MELS, N_FRAMES)

    with GradCAM(model) as cam_fn:
        cam_pos = cam_fn(x, target_sign=1.0)["cam"]
        cam_neg = cam_fn(x, target_sign=-1.0)["cam"]

    assert not np.allclose(cam_pos, cam_neg)


def test_gradcam_hooks_are_removed_on_close():
    model = CNNGroupNorm().eval()
    cam_fn = GradCAM(model)
    cam_fn.close()
    # After closing, the target layer should have no remaining hooks from us.
    assert len(cam_fn.target_layer._forward_hooks) == 0
    assert len(cam_fn.target_layer._backward_hooks) == 0


def test_summarize_cam_reports_localized_hot_region():
    # A synthetic CAM with a single sharp hot spot should report a narrow
    # hz_range/time_range and low normalized entropy (localized), not the
    # full spectrogram extent.
    cam = np.zeros((N_MELS, N_FRAMES), dtype=np.float32)
    cam[10:12, 50:55] = 1.0  # small localized hot region

    mel_freqs = np.linspace(0, 8000, N_MELS)
    summary = summarize_cam(cam, mel_freqs, hop_length=512, sample_rate=16000, threshold=0.7)

    assert summary["frac_hot"] < 0.01
    assert summary["normalized_entropy"] < 0.5
    assert summary["hz_range"][0] <= summary["hz_range"][1]


def test_summarize_cam_reports_diffuse_uniform_region_as_high_entropy():
    cam = np.ones((N_MELS, N_FRAMES), dtype=np.float32)
    mel_freqs = np.linspace(0, 8000, N_MELS)
    summary = summarize_cam(cam, mel_freqs, hop_length=512, sample_rate=16000, threshold=0.7)

    assert summary["normalized_entropy"] > 0.99
    assert summary["frac_hot"] == 1.0
