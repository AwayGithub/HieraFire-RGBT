# HieraFire

Code for **HieraFire: Hierarchical RGB-Thermal Feature Interaction for Real-Time UAV Wildfire Detection**.

HieraFire matches cross-modal fusion to feature hierarchy on dual YOLOv8n streams:
DMG at P2, CFRA at P3, and RS-SQF at P4/P5. Detection uses four scales;
unimodal P3 auxiliary heads provide training-only supervision.

## Installation

Python 3.11 and PyTorch 2.2.2 are the reference environment.

```bash
conda create -n hierafire python=3.11
conda activate hierafire
pip install -r requirements.txt
pip install -e .
```

The bundled `ultralytics` package contains the custom model and paired loader.
Do not replace it with the stock PyPI package. FlashAttention is optional;
the implementation provides a PyTorch fallback.

## Data

Obtain RGBT-3M or RGB-T Wildfire from their respective authors. Dataset images,
labels, and model weights are not distributed in this repository.
Prepare aligned visible/thermal pairs with matching filenames:

```text
DATA_ROOT/
  train.txt
  val.txt
  RGB/train/   RGB/val/
  IR/train/    IR/val/
  labels_fire_person/train/   labels_fire_person/val/
```

Each split file contains one image stem per line, without extension.
Labels use YOLO normalized `class x_center y_center width height` format.
RGBT-3M uses JPG images and class IDs `0=fire, 1=person`.
Wildfire uses PNG images, `labels/` instead of `labels_fire_person/`, and `0=fire`.
The framework names its evaluation split `val`; put the intended held-out split there.
Edit the dataset YAML if your extensions or label directories differ.

Raw samples are concatenated as VIS_BGR | IR_BGR. The final Format operation
reverses all six channels to IR_RGB | VIS_RGB, matching the network branches.
Geometry is shared between modalities; HSV affects visible images only.

## Train

```bash
python train.py --data configs/rgbt3m.yaml --data-root /path/to/RGBT-3M \
  --imgsz 480 640 --epochs 200 --batch 16 --cls 0.1 --device 0
```

The release defaults to SGD, initial LR 0.01, seed 0, HSV and horizontal flips,
with Mosaic/MixUp/CopyPaste disabled. `--augmentation none` disables augmentation;
`--augmentation nonmosaic` adds paired geometric transforms.

The recorded Wildfire experiment used 512x640 inputs and cls=0.5:

```bash
python train.py --data configs/wildfire.yaml --data-root /path/to/RGB-T-Wildfire \
  --imgsz 512 640 --epochs 200 --batch 16 --cls 0.5 \
  --augmentation nonmosaic --device 0 --name hierafire_wildfire
```

Outputs go to `runs/detect/`. Resume with the same dataset options and
`--resume runs/detect/hierafire/weights/last.pt`.
The current loader and release settings do not guarantee bitwise reproduction
of historical experiments; scores depend on checkpoint, split and evaluation settings.

## Evaluate and Predict

```bash
python val.py --weights /path/to/checkpoint.pt --data configs/rgbt3m.yaml \
  --data-root /path/to/RGBT-3M --imgsz 480 640 --device 0
python predict.py --weights /path/to/checkpoint.pt \
  --rgb /path/to/visible.jpg --ir /path/to/thermal.jpg --device cuda:0
```

Validation reports overall and per-class metrics. Pair prediction writes a
separate RGB and IR image with boxes and confidence scores under `runs/predict/`.
Only load checkpoints from trusted sources. Pretrained checkpoints are not
included in this release; train locally to produce them.

## Reported Results

| Dataset | P (%) | mAP50 (%) | mAP50-95 (%) |
| --- | ---: | ---: | ---: |
| RGBT-3M | 93.36 | 94.05 | 59.45 |
| RGB-T Wildfire | 93.48 | 91.26 | 50.08 |

Reported RGBT-3M complexity: 5.22M parameters, 13.49 GFLOPs,
21.19 ms forward latency (RTX 2080 Ti, batch 1, FP32).

## Code and Tests

- `configs/hierafire.yaml`: final architecture.
- `ultralytics/nn/tasks.py`: dual-stream model and auxiliary supervision.
- `ultralytics/nn/modules/`: fusion operators and framework layers.
- `ultralytics/data/dataset.py`: paired loading and augmentation.

```bash
python -m pytest tests -q -o addopts=
```

## License and Acknowledgments

This repository derives from YOLOv12 and Ultralytics and retains the AGPL-3.0
license and upstream notices. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
Research notes, manuscript sources, literature collections, external baseline
repositories, intermediate configurations and training outputs are excluded.
