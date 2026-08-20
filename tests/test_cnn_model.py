import torch

from src.cnn_model import SmallCNN


def test_forward_pass_shape():
    model = SmallCNN()
    x = torch.randn(4, 1, 64, 313)
    out = model(x)
    assert out.shape == (4,)


def test_forward_pass_single_sample():
    model = SmallCNN()
    x = torch.randn(1, 1, 64, 313)
    out = model(x)
    assert out.shape == (1,)
