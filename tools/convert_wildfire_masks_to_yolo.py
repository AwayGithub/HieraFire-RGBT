# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Convert RGB-Thermal Wildfire binary masks into fire-only YOLO bounding boxes."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def mask_to_yolo(mask: np.ndarray, class_id: int = 0, threshold: int = 127, min_area: int = 2) -> list[str]:
    """Convert each 8-connected foreground component into one normalized YOLO box."""
    if mask.ndim != 2:
        raise ValueError(f"Expected a single-channel mask, got shape {mask.shape}")
    height, width = mask.shape
    if height == 0 or width == 0:
        raise ValueError("Mask dimensions must be nonzero")

    binary = (mask > threshold).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    labels = []
    for component in range(1, count):
        x, y, box_width, box_height, area = (int(value) for value in stats[component])
        if area < min_area:
            continue
        center_x = (x + box_width / 2) / width
        center_y = (y + box_height / 2) / height
        labels.append(
            f"{class_id} {center_x:.8f} {center_y:.8f} {box_width / width:.8f} {box_height / height:.8f}"
        )
    return labels


def convert_masks(
    mask_dir: Path,
    output_dir: Path,
    class_id: int = 0,
    threshold: int = 127,
    min_area: int = 2,
    overwrite: bool = False,
) -> tuple[int, int, int]:
    """Convert all PNG masks recursively while preserving their relative directory structure."""
    masks = sorted(path for path in mask_dir.rglob("*.png") if path.is_file())
    if not masks:
        raise FileNotFoundError(f"No PNG masks found under: {mask_dir}")

    targets = [output_dir / path.relative_to(mask_dir).with_suffix(".txt") for path in masks]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {len(existing)} label files; pass --overwrite to replace them")

    total_boxes = 0
    empty_masks = 0
    for mask_path, target_path in zip(masks, targets):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Unable to read mask: {mask_path}")
        labels = mask_to_yolo(mask, class_id=class_id, threshold=threshold, min_area=min_area)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        total_boxes += len(labels)
        empty_masks += not labels
    return len(masks), total_boxes, empty_masks


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-dir", type=Path, required=True, help="source directory containing PNG masks")
    parser.add_argument("--output-dir", type=Path, required=True, help="destination directory for YOLO TXT labels")
    parser.add_argument("--class-id", type=int, default=0, help="fire class ID")
    parser.add_argument("--threshold", type=int, default=127, help="foreground threshold in [0, 255]")
    parser.add_argument("--min-area", type=int, default=2, help="minimum connected-component area in pixels")
    parser.add_argument("--overwrite", action="store_true", help="replace existing label files")
    return parser.parse_args()


def main() -> None:
    """Run mask conversion and print a compact summary."""
    args = parse_args()
    if not 0 <= args.threshold <= 255:
        raise ValueError(f"Threshold must be in [0, 255], got {args.threshold}")
    if args.class_id < 0:
        raise ValueError(f"Class ID must be nonnegative, got {args.class_id}")
    if args.min_area < 1:
        raise ValueError(f"Minimum area must be positive, got {args.min_area}")

    masks, boxes, empty_masks = convert_masks(
        args.mask_dir,
        args.output_dir,
        class_id=args.class_id,
        threshold=args.threshold,
        min_area=args.min_area,
        overwrite=args.overwrite,
    )
    print(f"Converted {masks} masks into {boxes} boxes; {empty_masks} masks produced empty label files.")


if __name__ == "__main__":
    main()
