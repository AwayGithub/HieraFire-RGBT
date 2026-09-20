# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Draw HieraFire predictions on one aligned RGB/thermal pair."""
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression


def paired_tensor(rgb, ir, shape):
    height, width = shape
    pair = np.concatenate([cv2.resize(rgb, (width, height)), cv2.resize(ir, (width, height))], axis=2)
    # Match Format: VIS_BGR | IR_BGR -> IR_RGB | VIS_RGB.
    return torch.from_numpy(np.ascontiguousarray(pair[..., ::-1].transpose(2, 0, 1))).float().unsqueeze(0) / 255


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--ir", required=True)
    parser.add_argument("--imgsz", nargs=2, type=int, default=[480, 640])
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=Path("runs/predict"))
    args = parser.parse_args()
    rgb, ir = cv2.imread(args.rgb), cv2.imread(args.ir)
    if rgb is None or ir is None:
        raise FileNotFoundError("Both --rgb and --ir must be readable images")
    if rgb.shape[:2] != ir.shape[:2]:
        raise ValueError("Input modalities must have matching, aligned image dimensions")
    model = YOLO(args.weights).model.to(args.device).float().eval()
    with torch.inference_mode():
        prediction = model(paired_tensor(rgb, ir, args.imgsz).to(args.device))
        boxes = non_max_suppression(prediction, args.conf, 0.7, nc=len(model.names))[0].cpu()
    args.output.mkdir(parents=True, exist_ok=True)
    for modality, canvas in [("rgb", rgb.copy()), ("ir", ir.copy())]:
        for x1, y1, x2, y2, confidence, cls in boxes.tolist():
            p1 = (round(x1 * canvas.shape[1] / args.imgsz[1]), round(y1 * canvas.shape[0] / args.imgsz[0]))
            p2 = (round(x2 * canvas.shape[1] / args.imgsz[1]), round(y2 * canvas.shape[0] / args.imgsz[0]))
            cv2.rectangle(canvas, p1, p2, (0, 200, 255), 2)
            cv2.putText(canvas, f"{model.names[int(cls)]} {confidence:.2f}", p1,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.imwrite(str(args.output / f"{modality}.jpg"), canvas)


if __name__ == "__main__":
    main()
