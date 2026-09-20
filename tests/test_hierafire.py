# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
from pathlib import Path
import numpy as np
import torch
from ultralytics.nn.tasks import DualStreamDetectionModel
from predict import paired_tensor


def test_modality_order():
    rgb = np.full((32, 32, 3), [10, 20, 30], dtype=np.uint8)
    ir = np.full((32, 32, 3), [40, 50, 60], dtype=np.uint8)
    actual = paired_tensor(rgb, ir, [32, 32])[0, :, 0, 0]
    torch.testing.assert_close(actual, torch.tensor([60, 50, 40, 30, 20, 10]) / 255)


def test_model_forward_and_backward():
    torch.set_num_threads(1)
    cfg = str(Path(__file__).resolve().parents[1] / "configs/hierafire.yaml")
    model = DualStreamDetectionModel(cfg, nc=2, verbose=False)
    assert set(model.fusion_convs) == {"p2", "p3", "p4", "p5"}
    images = torch.rand(2, 6, 64, 96)
    model.train()
    output = model(images)
    loss = sum(x.square().mean() for x in output)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.fusion_convs.parameters())
    model.eval()
    with torch.inference_mode():
        prediction = model(images)[0]
    assert prediction.shape[0:2] == (2, 6)
    assert torch.isfinite(prediction).all()
