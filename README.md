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

Both datasets are distributed through the USTC Complex Laboratory
[dataset page](https://complex.ustc.edu.cn/sjwwataset/list.htm):

- **RGBT-3M** contains 11,220 registered RGB-thermal pairs at 640x480.
- **RGB-Thermal Wildfire** contains 1,367 paired images at 420x420 or 640x512.

Please follow the download and citation instructions on the official page.
Dataset images, labels, and model weights are not distributed in this repository.
The exact sample lists used for the reported experiments are included under
`splits/`:

| Dataset | Training pairs | Held-out test pairs | Split files |
| --- | ---: | ---: | --- |
| RGBT-3M | 7,854 | 3,366 | `splits/rgbt3m/` |
| RGB-Thermal Wildfire | 1,162 | 205 | `splits/rgbt_wildfire/` |

Copy the appropriate lists into the prepared dataset root:

```bash
cp splits/rgbt3m/{train,val}.txt /path/to/RGBT-3M/
cp splits/rgbt_wildfire/{train,val}.txt /path/to/RGB-T-Wildfire/
```

RGB-Thermal Wildfire provides pixel masks rather than detection boxes. After
any spatial resizing or padding, convert those masks into the fire-only YOLO
labels used here:

```bash
python tools/convert_wildfire_masks_to_yolo.py \
  --mask-dir /path/to/RGB-T-Wildfire/masks \
  --output-dir /path/to/RGB-T-Wildfire/labels
```

The converter creates one box per 8-connected foreground component, assigns
class ID `0` (fire), and removes isolated one-pixel mask noise. Empty masks
produce empty label files. Run it on masks after geometric preprocessing so
their dimensions remain identical to the paired RGB and thermal images.

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
For loader compatibility, the held-out test list is named `val.txt` and assigned
to the `val` field. It is evaluated every epoch for metric logging only and is
not used for early stopping or checkpoint selection.
Edit the dataset YAML if your extensions or label directories differ.

Raw samples are concatenated as VIS_BGR | IR_BGR. The final Format operation
reverses all six channels to IR_RGB | VIS_RGB, matching the network branches.
Geometry is shared between modalities; HSV affects visible images only.

## Train

```bash
python train.py --data configs/rgbt3m.yaml --data-root /path/to/RGBT-3M \
  --imgsz 480 640 --epochs 200 --batch 16 --cls 0.1 --device 0
```

The release defaults to SGD, initial LR 0.01, HSV and horizontal flips, with
Mosaic/MixUp/CopyPaste disabled. A fresh random seed is generated for each run;
pass `--seed N` to reproduce a specific run. `--augmentation none` disables
augmentation; `--augmentation nonmosaic` adds paired geometric transforms.

The recorded Wildfire experiment used 512x640 inputs and cls=0.5:

```bash
python train.py --data configs/wildfire.yaml --data-root /path/to/RGB-T-Wildfire \
  --imgsz 512 640 --epochs 200 --batch 16 --cls 0.5 \
  --augmentation nonmosaic --device 0 --name hierafire_wildfire
```

Outputs go to `runs/detect/`. Resume with the same dataset options and
`--resume runs/detect/hierafire/weights/last.pt`.
Training runs for the requested epoch count (200 by default) without early
stopping. The held-out test split is evaluated every epoch only to record
metrics; it does not select a `best.pt` checkpoint. `last.pt` and every
`epoch*.pt` are saved, and final results should be reported from the completed
final epoch.
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

Reproduce the parameter and FLOPs measurement with a real six-channel,
full-resolution forward pass:

```bash
python profile_flops.py --model configs/hierafire.yaml --imgsz 480 640 --batch 1
```

The script uses PyTorch profiler operator counts. One multiplication and one
addition count as two FLOPs.

## Code and Tests

- `configs/hierafire.yaml`: final architecture.
- `profile_flops.py`: parameter and full-input FLOPs measurement.
- `tools/convert_wildfire_masks_to_yolo.py`: Wildfire mask-to-box conversion.
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
