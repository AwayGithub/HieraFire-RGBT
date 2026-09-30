# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Validate a trusted HieraFire checkpoint on the configured evaluation split."""
import argparse
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import yaml_load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="configs/rgbt3m.yaml")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, nargs=2, default=[480, 640])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="hierafire_val")
    args = parser.parse_args()
    data = yaml_load(args.data)
    data["path"] = str(args.data_root.expanduser().resolve())
    data["img_size"] = args.imgsz
    YOLO(args.weights).val(data=data, imgsz=args.imgsz, batch=args.batch,
                          device=args.device, half=False, plots=True,
                          project=str(Path("runs/detect").resolve()), name=args.name)


if __name__ == "__main__":
    main()
