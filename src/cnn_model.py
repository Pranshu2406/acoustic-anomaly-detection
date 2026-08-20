"""CNNs for log-Mel spectrogram anomaly classification.

Both variants are kept deliberately small: ~4,100 training clips per
leave-one-machine-out fold doesn't support a deep architecture, the same
complexity-discipline reasoning behind the classical baselines in
src/baselines.py.

CNNGroupNorm is the default/primary model (see src/train_cnn.py). SmallCNN
(BatchNorm) is kept only as a comparison/ablation baseline -- BatchNorm's
running mean/var are fit on the training machines only, and under the
distribution shift observed between MIMII machines (some run louder / have a
compressed dynamic range relative to others -- see README), those frozen
eval-time statistics collapsed predictions on 2 of 4 held-out machines
(id_02, id_06) to below-chance PR-AUC. Swapping to GroupNorm, with everything
else identical, resolved both: mean PR-AUC across folds went from 0.410 to
0.674, and every fold moved above its own chance baseline. Use SmallCNN only
when deliberately re-running that ablation (scripts/compare_cnn_norm.py).
"""

import torch
import torch.nn as nn


def _num_groups(channels: int, preferred: int = 8) -> int:
    """Largest divisor of `channels` that is <= `preferred` -- GroupNorm requires
    num_groups to evenly divide the channel count."""
    for g in range(min(preferred, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


def _group_norm_conv_block(in_channels: int, out_channels: int, preferred_groups: int = 8) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(_num_groups(out_channels, preferred_groups), out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CNNGroupNorm(nn.Module):
    """Default CNN: 3 conv+groupnorm+relu+maxpool blocks, global average
    pooling, small FC head.

    Input: (batch, 1, n_mels, n_frames). Output: (batch,) raw logits -- pair
    with BCEWithLogitsLoss (which applies the sigmoid internally) rather than
    a separate sigmoid layer.
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            _group_norm_conv_block(in_channels, 16),
            _group_norm_conv_block(16, 32),
            _group_norm_conv_block(32, 64),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x.squeeze(1)


def _batch_norm_conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """BatchNorm ablation/comparison model -- NOT the default (see module
    docstring). Identical architecture to CNNGroupNorm except BatchNorm2d in
    place of GroupNorm.
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            _batch_norm_conv_block(in_channels, 16),
            _batch_norm_conv_block(16, 32),
            _batch_norm_conv_block(32, 64),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x.squeeze(1)


# The default CNN architecture used by src/train_cnn.py. SmallCNN (BatchNorm)
# remains available for explicit comparison/ablation runs only.
DEFAULT_CNN = CNNGroupNorm
