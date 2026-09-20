# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Train HieraFire on paired RGB-thermal images."""
import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
from ultralytics import YOLO
from ultralytics.utils import yaml_load


def configure_auxiliary_losses(trainer):
    trainer.model.aux_loss_weight = 0.25
    trainer.model.use_aux_head = True
    trainer.loss_names = ("box_loss", "cls_loss", "dfl_loss", "aux_rgb", "aux_ir")
    keys = trainer.validator.results_csv_keys() + trainer.label_loss_items(prefix="val")
    trainer.metrics = dict.fromkeys(keys, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="configs/rgbt3m.yaml")
    parser.add_argument("--model", default="configs/hierafire.yaml")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, nargs=2, default=[480, 640], metavar=("H", "W"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--cls", type=float, default=0.1)
    parser.add_argument("--name", default="hierafire")
    parser.add_argument("--resume", type=str)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--augmentation", choices=["hsv-flip", "nonmosaic", "none"], default="hsv-flip")
    args = parser.parse_args()
    cv2.setNumThreads(0)
    data = yaml_load(args.data)
    data["path"] = str(args.data_root.expanduser().resolve())
    data["img_size"] = args.imgsz
    model = YOLO(args.resume or args.model)
    model.add_callback("on_train_start", configure_auxiliary_losses)
    augmentation = dict(mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=0.0,
                        translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
                        flipud=0.0, fliplr=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4)
    if args.augmentation == "none":
        augmentation.update(fliplr=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0)
    elif args.augmentation == "nonmosaic":
        augmentation.update(hsv_h=0.005, hsv_s=0.2, hsv_v=0.2, degrees=8.0,
                            translate=0.05, scale=0.15, shear=2.0,
                            perspective=0.0003, flipud=0.1)
    model.train(data=data, resume=bool(args.resume), epochs=args.epochs, batch=args.batch,
                imgsz=args.imgsz, device=args.device, workers=args.workers,
                seed=0, deterministic=args.deterministic, optimizer="SGD", lr0=0.01,
                lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0,
                warmup_momentum=0.8, warmup_bias_lr=0.0, cls=args.cls, dfl=1.5,
                cos_lr=False, pretrained=False, amp=not args.no_amp, val_period=1,
                project=str(Path("runs/detect").resolve()), name=args.name, **augmentation)


if __name__ == "__main__":
    main()
