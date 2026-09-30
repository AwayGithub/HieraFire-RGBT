# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from profile_flops import profile_flops
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import FLAME2Dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.tasks import DualStreamDetectionModel
from ultralytics.utils.torch_utils import get_flops

from predict import paired_tensor


def test_modality_order():
    rgb = np.full((32, 32, 3), [10, 20, 30], dtype=np.uint8)
    ir = np.full((32, 32, 3), [40, 50, 60], dtype=np.uint8)
    actual = paired_tensor(rgb, ir, [32, 32])[0, :, 0, 0]
    torch.testing.assert_close(actual, torch.tensor([60, 50, 40, 30, 20, 10]) / 255)


def test_wildfire_dataset_loads_documented_png_layout(tmp_path):
    rgb_train = tmp_path / "RGB" / "train"
    ir_train = tmp_path / "IR" / "train"
    label_train = tmp_path / "labels" / "train"
    for directory in (rgb_train, ir_train, label_train):
        directory.mkdir(parents=True)

    image = np.full((32, 48, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(rgb_train / "1A_gt.png"), image)
    assert cv2.imwrite(str(ir_train / "1A_gt.png"), image)
    (label_train / "1A_gt.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    split = tmp_path / "train.txt"
    split.write_text("1A_gt\n", encoding="utf-8")

    data = {
        "path": str(tmp_path),
        "rgb_dir": "RGB",
        "thermal_dir": "IR",
        "label_dir": "labels",
        "image_ext": ".png",
        "img_size": [32, 48],
        "input_mode": "dual_input",
        "names": {0: "fire"},
    }
    dataset = FLAME2Dataset(
        img_path=split,
        imgsz=[32, 48],
        batch_size=1,
        augment=False,
        hyp=get_cfg(),
        data=data,
    )

    assert dataset.im_files == [str(rgb_train / "1A_gt.png")]
    sample = dataset[0]
    assert sample["img"].shape == (6, 32, 48)
    assert sample["cls"].tolist() == [[0.0]]


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


def test_yaml_builds_dual_stream_model_and_profiles_flops():
    torch.set_num_threads(1)
    cfg = Path(__file__).resolve().parents[1] / "configs/hierafire.yaml"
    model = YOLO(str(cfg)).model
    assert isinstance(model, DualStreamDetectionModel)
    assert model.ch == 6
    assert sum(parameter.numel() for parameter in model.parameters()) == 5_219_858
    flops, breakdown = profile_flops(model, (1, 6, 480, 640), torch.device("cpu"))
    assert flops / 1e9 == pytest.approx(13.49, abs=0.01)
    assert breakdown["aten::conv2d"] > 0
    assert get_flops(model, [64, 96]) > 0


def test_fixed_epoch_training_defaults():
    cfg = get_cfg()
    assert cfg.epochs == 200
    assert cfg.patience == 0
    assert cfg.save_best is False
    assert cfg.save_period == 1


def test_checkpoint_policy_does_not_save_best(tmp_path):
    model = torch.nn.Linear(1, 1)
    trainer = SimpleNamespace(
        epoch=0,
        best_fitness=0.5,
        fitness=0.5,
        metrics={},
        ema=SimpleNamespace(ema=model, updates=0),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        args=SimpleNamespace(save_best=False),
        last=tmp_path / "last.pt",
        best=tmp_path / "best.pt",
        wdir=tmp_path,
        save_period=1,
        collect_model_debug_metrics=lambda: {},
        read_results_csv=lambda: {},
    )

    BaseTrainer.save_model(trainer, save_val_snapshot=True)

    assert trainer.last.exists()
    assert (tmp_path / "epoch0.pt").exists()
    assert not trainer.best.exists()
