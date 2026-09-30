# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
import cv2
import numpy as np

from tools.convert_wildfire_masks_to_yolo import convert_masks, mask_to_yolo


def test_mask_to_yolo_uses_components_and_filters_single_pixel_noise():
    mask = np.zeros((20, 40), dtype=np.uint8)
    mask[2:6, 4:12] = 255
    mask[10:18, 24:36] = 255
    mask[0, 0] = 255

    assert mask_to_yolo(mask) == [
        "0 0.20000000 0.20000000 0.20000000 0.20000000",
        "0 0.75000000 0.70000000 0.30000000 0.40000000",
    ]


def test_convert_masks_preserves_subdirectories_and_empty_labels(tmp_path):
    mask_dir = tmp_path / "masks"
    output_dir = tmp_path / "labels"
    (mask_dir / "train").mkdir(parents=True)
    (mask_dir / "val").mkdir(parents=True)

    positive = np.zeros((10, 10), dtype=np.uint8)
    positive[2:5, 3:7] = 255
    cv2.imwrite(str(mask_dir / "train" / "positive.png"), positive)
    cv2.imwrite(str(mask_dir / "val" / "empty.png"), np.zeros((10, 10), dtype=np.uint8))

    assert convert_masks(mask_dir, output_dir) == (2, 1, 1)
    assert (output_dir / "train" / "positive.txt").read_text() == "0 0.50000000 0.35000000 0.40000000 0.30000000\n"
    assert (output_dir / "val" / "empty.txt").read_text() == ""
