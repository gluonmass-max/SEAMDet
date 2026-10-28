> **论文发表后开源** — The code will be released after the paper is published.

# SEAMDet: Stitching a SAR-Domain Expert into an Early-Fusion Detector for Optical–SAR Oriented Object Detection

This repository contains the implementation of SEAMDet, an oriented object detector for paired optical and SAR images. It is built on MMRotate 1.x (OpenMMLab).

## Overview

SEAMDet extends a single-backbone early-fusion detector. The optical (RGB) and SAR images are stacked into one 4-channel input, and a SAR-domain expert is attached to the backbone. 

## Requirements

- Linux, Python 3.10
- PyTorch ≥ 2.4 with CUDA. Tested with PyTorch 2.4 (CUDA 12.1) and PyTorch 2.7 (CUDA 12.8).
- MMEngine, MMCV 2.x (with CUDA ops), MMDetection 3.x, MMRotate 1.x
- `shapely`, `numpy`, `opencv-python` for evaluation and data tools

## Installation

```bash
git clone <repo-url> SEAMDet && cd SEAMDet
conda create -n seamdet python=3.10 -y && conda activate seamdet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # pick the wheel matching your CUDA
pip install -U openmim && mim install mmengine "mmcv>=2.0.0" "mmdet>=3.0.0"
pip install -v -e .                       # installs the bundled MMRotate 1.x code base
pip install shapely opencv-python
```

## Repository layout

```
projects/SEAMDet/
  seamdet/           dataset, paired optical/SAR loading, 4-channel data preprocessor, backbones (LSK-EF + SESF,
                     ResNet-50 + SESF), RPN head, prediction dump metric
  configs/           SEAMDet, the SAR-only expert detector, early-fusion baselines and ablation variants
tools/               train.py / test.py (MMEngine entry points)
tools/seamdet/       pretrained-weight conversion, dataset conversion, evaluation
```

## Data preparation

Both datasets use the same on-disk layout. Each image id has an optical image, a co-registered SAR image, and one label file. The label format is Ultralytics OBB: `class x1 y1 x2 y2 x3 y3 x4 y4`, with coordinates normalised to [0, 1].

```
data/<DATASET>/
  optical/images/{train,val,test}/<id>.<ext>
  sar/images/{train,val,test}/<id>.<ext>
  optical/labels/{train,val,test}/<id>.txt
```

| Dataset       | Image size | Preparation                                                  |
| ------------- | ---------- | ------------------------------------------------------------ |
| **M4-SAR**    | 512 × 512  | The official release already uses this layout. Point `data_root` in the config to it. |
| **OGSOD-2.0** | 256 × 256  | Convert the official release with `python tools/seamdet/make_ogsod.py <ogsod_root> data/OGSOD-2.0`. |

**Test subsets.** Evaluation can be restricted to named subsets of the test images, given as a JSON file `{name: [image ids]}`. We use this to exclude test images whose annotation files are byte-identical (MD5) to a training image. The subset files are provided under `tools/seamdet/subsets/`.

## Pretrained weights

The backbones start from ImageNet-pretrained weights. Their first convolution is extended to 4 input channels; the extra SAR channel is initialised with the mean of the RGB filters:

```bash
python tools/seamdet/make_in4.py lsk_s_backbone.pth weights/lsk_s_backbone_in4.pth
```

## Training

SEAMDet is trained in two stages. Both stages use the same schedule: 50 epochs, 512 × 512 input, batch size 16, AdamW (lr 2e-4, weight decay 0.05), linear warm-up followed by cosine decay, mixed precision, EMA, and gradient clipping.

**Stage 1 — SAR-only expert detector** (LSKNet-S on the SAR image):

```bash
python tools/train.py projects/SEAMDet/configs/sar_only_lsk-s_m4sar.py --work-dir work_dirs/sar_only_m4sar
```

**Stage 2 — SEAMDet.** The SESF expert stream is initialised from the Stage-1 checkpoint (SDI). Set `model.backbone.ms_init_ckpt` to it:

```bash
python tools/train.py projects/SEAMDet/configs/seamdet_lsk-s_m4sar.py --work-dir work_dirs/seamdet_m4sar \
    --cfg-options model.backbone.ms_init_ckpt=work_dirs/sar_only_m4sar/epoch_50.pth
```

**Other options**

- Use the `*_ogsod.py` configs for OGSOD-2.0.
- Set a seed with `--cfg-options randomness.seed=<seed>`.
- For multi-GPU training, use `tools/dist_train.sh <config> <num_gpus>`.

## Testing

Run inference on the test set; the predictions are written to a JSON file. Two-stage detectors are tested with up to 500 RPN proposals per image:

```bash
python tools/test.py projects/SEAMDet/configs/seamdet_lsk-s_m4sar.py work_dirs/seamdet_m4sar/epoch_50.pth \
    --work-dir work_dirs/seamdet_m4sar/test \
    --cfg-options model.test_cfg.rpn.max_per_img=500 test_evaluator.outfile=work_dirs/seamdet_m4sar/predictions_test.json
```

Each entry of the prediction file has the form `{"image_id", "category_id" (0-based), "score", "poly": [x1, y1, ..., x4, y4]}`, in pixels.

## Evaluation

The evaluators read the prediction JSON and the ground-truth label files, so they can score detectors from any framework.

**Unified evaluator** (polygon IoU):

```bash
python tools/seamdet/eval_unified.py --pred predictions_test.json \
    --labels data/M4-SAR/optical/labels/test --subsets tools/seamdet/subsets/m4sar_test.json \
    --size 512 --out results_unified.json
```

- IoU is the exact polygon IoU between the predicted box and the ground-truth quadrilateral.
- It reports COCO-style AP50, AP75 and AP50:95 per class and per subset, a VOC07 AP50, and AP by object size (tiny / small / medium+).

**Official-style evaluator** (ProbIoU mAP, following the M4-SAR protocol):

```bash
python tools/seamdet/eval_official.py --pred predictions_test.json \
    --labels data/M4-SAR/optical/labels/test --subsets tools/seamdet/subsets/m4sar_test.json --size 512
```

**OGSOD-2.0:** pass the OGSOD-2.0 label directory and subset file, with `--size 256`.

## Model complexity

Parameters and GMACs at a given input size and number of RPN proposals:

```bash
python tools/seamdet/complexity.py projects/SEAMDet/configs/seamdet_lsk-s_m4sar.py --size 512 --proposals 500
```

## Citation

```bibtex
@article{seamdet,
  title   = {SEAMDet: Stitching a SAR-Domain Expert into an Early-Fusion Detector for Optical--SAR Oriented Object Detection},
  author  = {},
  journal = {},
  year    = {}
}
```

## Acknowledgements

This code builds on [MMRotate](https://github.com/open-mmlab/mmrotate) and [LSKNet](https://github.com/zcablii/LSKNet). We thank the authors of the M4-SAR and OGSOD-2.0 datasets for making their data available.

## License

To be announced with the code release.
