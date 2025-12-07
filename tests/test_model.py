import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.model import build_model


def test_forward_shapes():
    model = build_model(image_size=(256, 256), num_classes=5)
    x = torch.randn(2, 1, 256, 256)
    out = model(x)
    assert out["logits"].shape == (2, 5)
    assert out["homography"].shape == (2, 3, 3)
    assert out["warped"].shape[2:] == model.frame.patch_size
