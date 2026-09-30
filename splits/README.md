# Dataset Splits

These are the exact sample lists used for the reported HieraFire experiments.
Each line is an image stem without a filename extension. RGB and thermal files
with the same stem form one registered pair.

| Directory | Train | Held-out test | Total |
| --- | ---: | ---: | ---: |
| `rgbt3m/` | 7,854 | 3,366 | 11,220 |
| `rgbt_wildfire/` | 1,162 | 205 | 1,367 |

The evaluation files retain the name `val.txt` because the paired Ultralytics
loader consumes its held-out split through the `val` configuration field. The
public training entry point logs metrics on this split but does not use them for
early stopping or checkpoint selection.

Download both datasets from the USTC Complex Laboratory
[dataset page](https://complex.ustc.edu.cn/sjwwataset/list.htm), then copy the
corresponding `train.txt` and `val.txt` files into the prepared dataset root.
