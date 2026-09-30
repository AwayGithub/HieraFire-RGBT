# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Measure HieraFire parameters and FLOPs with a real six-channel forward pass."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.nn.tasks import DualStreamDetectionModel


def load_profile_model(source: Path) -> torch.nn.Module:
    """Load YAML or checkpoint and rebuild serialized dual-stream models for compatibility."""
    loaded = YOLO(str(source)).model
    yaml = getattr(loaded, "yaml", {})
    if source.suffix.lower() == ".pt" and yaml.get("dual_stream"):
        model = DualStreamDetectionModel(deepcopy(yaml), ch=6, nc=yaml.get("nc"), verbose=False)
        model.load_state_dict(loaded.state_dict(), strict=True)
        return model
    return loaded


def profile_flops(
    model: torch.nn.Module,
    input_shape: tuple[int, int, int, int],
    device: torch.device,
) -> tuple[int, Counter]:
    """Return profiler-supported FLOPs and an operator-level breakdown."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    model = model.to(device=device, dtype=torch.float32).eval()
    sample = torch.zeros(input_shape, device=device, dtype=torch.float32)
    with torch.inference_mode(), torch.profiler.profile(activities=activities, with_flops=True) as profiler:
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    breakdown = Counter()
    for event in profiler.key_averages():
        flops = int(event.flops or 0)
        if flops:
            breakdown[event.key] += flops
    return sum(breakdown.values()), breakdown


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("configs/hierafire.yaml"))
    parser.add_argument("--imgsz", type=int, nargs=2, default=(480, 640), metavar=("H", "W"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0")
    parser.add_argument("--topk", type=int, default=6, help="number of operator types to print")
    return parser.parse_args()


def main() -> None:
    """Load a model, run the profiler, and print a reproducible complexity report."""
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model configuration or checkpoint not found: {args.model}")
    if args.batch < 1:
        raise ValueError(f"Batch size must be positive, got {args.batch}")

    device = torch.device(args.device)
    model = load_profile_model(args.model)
    channels = int(getattr(model, "ch", 3))
    if args.model.suffix.lower() in {".yaml", ".yml"} and not isinstance(model, DualStreamDetectionModel):
        raise RuntimeError("A dual-stream YAML must construct DualStreamDetectionModel, but it did not.")

    input_shape = (args.batch, channels, args.imgsz[0], args.imgsz[1])
    total_flops, breakdown = profile_flops(model, input_shape, device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    print(f"Model: {args.model}")
    print(f"Model class: {type(model).__name__}")
    print(f"Input: {input_shape} (FP32)")
    print(f"Parameters: {parameters:,} ({parameters / 1e6:.4f} M)")
    print(f"Trainable parameters: {trainable:,} ({trainable / 1e6:.4f} M)")
    print(f"FLOPs: {total_flops:,} ({total_flops / 1e9:.4f} GFLOPs)")
    print("Convention: one multiplication and one addition count as two FLOPs.")
    print("Operator coverage follows PyTorch profiler's with_flops support.")
    if args.topk > 0:
        print("Top operators:")
        for operator, flops in breakdown.most_common(args.topk):
            print(f"  {operator}: {flops / 1e9:.6f} GFLOPs")


if __name__ == "__main__":
    main()
