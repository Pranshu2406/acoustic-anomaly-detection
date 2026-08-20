"""Grad-CAM for CNNGroupNorm.

Diagnostic goal (see README "Grad-CAM interpretability"), beyond general
interpretability: determine whether the model attends to localized,
anomaly-consistent time-frequency regions, or to diffuse/broadband patterns
that would suggest it's exploiting a session-level recording confound of the
kind uncovered during the calibration-leakage investigation (src/calibration.py).

Target layer: the ReLU output of the last conv block (`features[2][2]`) --
the finest-resolution post-nonlinearity feature map that still feeds into the
model's global average pooling (before that block's own MaxPool2d further
downsamples it). This is the standard Grad-CAM choice of "last conv layer's
activations."

Grad-CAM's weighting (global-average-pooled gradients per channel) has no
BatchNorm-specific assumptions baked in -- it only needs activations and
gradients at the target layer, which GroupNorm provides identically to
BatchNorm in eval mode. That's still verified empirically rather than assumed:
tests/test_gradcam.py checks the produced CAMs are finite and non-constant.
"""

import numpy as np
import torch
import torch.nn.functional as F

from src.cnn_model import CNNGroupNorm


class GradCAM:
    """Grad-CAM for a single-logit binary classifier.

    `target_sign`, passed to `__call__`, controls which class's evidence is
    explained: +1 backpropagates the raw logit (rising logit = more
    "abnormal"), so the CAM shows what pushes toward "abnormal". -1
    backpropagates the negated logit, showing what pushes toward "normal" --
    there's no separate "normal" output to target directly since this model
    has one sigmoid output, not two class logits.
    """

    def __init__(self, model: CNNGroupNorm, target_layer: torch.nn.Module = None):
        self.model = model.eval()
        self.target_layer = target_layer if target_layer is not None else model.features[2][2]
        if isinstance(self.target_layer, torch.nn.ReLU) and self.target_layer.inplace:
            # register_full_backward_hook is incompatible with an in-place op on
            # the hooked module (PyTorch raises "is a view and is being modified
            # inplace"). Switching to out-of-place doesn't change the computed
            # values -- ReLU(x) is identical either way -- only how memory is reused.
            self.target_layer.inplace = False
        self._activations = None
        self._gradients = None
        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self._activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0]

    def __call__(self, x: torch.Tensor, target_sign: float = 1.0) -> dict:
        """x: (1, 1, n_mels, n_frames). Returns a dict with:
        - cam: (n_mels, n_frames) numpy array, min-max normalized to [0, 1],
          upsampled (bilinear) from the target layer's spatial resolution to
          match the input spectrogram.
        - logit, proba: the model's raw output for this input.
        """
        self.model.zero_grad(set_to_none=True)
        logit = self.model(x)  # shape (1,)
        (target_sign * logit).backward()

        activations = self._activations  # (1, C, H, W)
        gradients = self._gradients  # (1, C, H, W)
        if activations is None or gradients is None:
            raise RuntimeError("Grad-CAM hooks did not fire -- target_layer may not be part of the forward graph.")

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # global-average-pooled gradients, per channel
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))  # (1, 1, H, W)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().numpy()

        cam_min, cam_max = float(cam.min()), float(cam.max())
        if cam_max - cam_min > 1e-12:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return {"cam": cam, "logit": logit.item(), "proba": torch.sigmoid(logit).item()}

    def close(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def summarize_cam(cam: np.ndarray, mel_freqs: np.ndarray, hop_length: int, sample_rate: int, threshold: float = 0.7) -> dict:
    """Quantifies how localized (vs. diffuse) a min-max-normalized CAM is:

    - normalized_entropy: Shannon entropy of the CAM treated as a probability
      distribution over pixels, divided by the maximum possible entropy
      (uniform distribution). ~0 = all mass on a few pixels (localized);
      ~1 = spread evenly across the whole spectrogram (diffuse).
    - hz_range / time_range_s: the frequency/time extent of "hot" pixels
      (cam >= threshold) -- a concrete, reportable localization footprint.
    - frac_hot: fraction of pixels at or above threshold.
    """
    eps = 1e-12
    p = cam / (cam.sum() + eps)
    entropy = float(-np.sum(p * np.log(p + eps)))
    max_entropy = float(np.log(cam.size))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    hot = cam >= threshold
    frac_hot = float(hot.mean())
    if hot.any():
        mel_rows, time_cols = np.where(hot)
        hz_range = (float(mel_freqs[mel_rows.min()]), float(mel_freqs[mel_rows.max()]))
        time_range_s = (
            float(time_cols.min() * hop_length / sample_rate),
            float(time_cols.max() * hop_length / sample_rate),
        )
    else:
        hz_range = (float("nan"), float("nan"))
        time_range_s = (float("nan"), float("nan"))

    return {
        "normalized_entropy": normalized_entropy,
        "frac_hot": frac_hot,
        "hz_range": hz_range,
        "time_range_s": time_range_s,
    }
