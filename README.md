# Marine Debris Detection from Sentinel-2 Imagery
## Multi-Scale Spectral Index Fusion via DeepSpectralUNet

**Blekinge Institute of Technology — Deep Learning Project 2026**

| Author | Email |
|--------|-------|
| Saideep Reddy Tamma | satm25@student.bth.se |
| Sriya Chittaneni | srch25@student.bth.se |
| Meghashyam Sai Dontha | medo25@student.bth.se |

---

## Overview

This project implements and evaluates seven models for marine debris
semantic segmentation on the MARIDA benchmark dataset. We propose
**DeepSpectralUNet** — a novel encoder-decoder architecture that
processes FDI, NDVI, NDWI and FAI spectral indices through a dedicated
multi-scale branch fused at every decoder stage. The model is evaluated
against five baselines including Random Forest, U-Net, Attention U-Net,
DeepLabV3+ and ResAttUNet-CBAM.

---

## Models

| Model | Type | Parameters |
|-------|------|-----------|
| Random Forest | Classical baseline | N/A |
| U-Net | Deep learning baseline | 31M |
| Attention U-Net | Attention baseline | 31M |
| DeepLabV3+ | Dilated convolution baseline | 16M |
| ResAttUNet-CBAM | State of art Mohammed 2022 | 32.8M |
| SpectralAttUNet | Proposed single-stage fusion | 70.7M |
| DeepSpectralUNet | Proposed multi-scale fusion | 75.7M |

---

## Dataset

**MARIDA** — Marine Debris Archive (Kikaki et al. 2022)

- 1,381 Sentinel-2 patches of size 256×256 pixels
- 11 spectral bands + 4 computed indices = 15 channels total
- 10 merged output classes
- Splits: 694 train / 328 val / 359 test
- Marine debris = only 0.45% of all pixels

### Download Option 1 — Kaggle (faster):

```bash
pip install kaggle

# place kaggle.json in ~/.kaggle/ first
# get it from https://www.kaggle.com/settings → API → Create New Token

kaggle datasets download -d anangfath/marida-marine-debrish-dataset
unzip marida-marine-debrish-dataset.zip -d MARIDA
```

Kaggle link: https://www.kaggle.com/datasets/anangfath/marida-marine-debrish-dataset

### Download Option 2 — Zenodo (official):

```python
import urllib.request
import zipfile

print("Downloading MARIDA — 4.38 GB, this will take a while...")
urllib.request.urlretrieve(
    "https://zenodo.org/record/5151941/files/MARIDA.zip",
    "MARIDA.zip")

print("Extracting...")
with zipfile.ZipFile("MARIDA.zip", "r") as z:
    z.extractall(".")

print("Done. Dataset ready at MARIDA/")
```

Zenodo link: https://zenodo.org/record/5151941

### After downloading place files like this:

```
marine5/
├── MARIDA/
│   ├── patches/    ← satellite image patches
│   └── splits/     ← train_X.txt val_X.txt test_X.txt
```

---

## Project Structure

```
marine/
├── marine_debris.ipynb              ← main notebook
├── README.md
├── requirements.txt
├── outputs/
│   ├── checkpoints/           ← saved model weights
│   │   ├── unet_fixed_best.pth
│   │   ├── attunet_fixed_best.pth
│   │   ├── deeplabv3plus_best.pth
│   │   ├── resattunet_cbam_best.pth
│   │   ├── spectral_attunet_best.pth
│   │   └── deep_spectral_unet_best.pth
│   ├── results/               ← saved figures
│   │   ├── fig1_prediction_maps.png
│   │   ├── fig2_confusion_matrix.png
│   │   ├── fig3_per_class_iou.png
│   │   ├── fig4_attention_maps.png
│   │   ├── fig5_spectral_features.png
│   │   └── fig6_training_curves.png
│   └── train_stats.json       ← normalisation statistics
└── MARIDA/
    ├── patches/               ← satellite patches
    └── splits/                ← train/val/test splits
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and CUDA 12.0+

---

## Running the Notebook

Open `marine_debris.ipynb` in Jupyter:

```bash
jupyter notebook marine_debris.ipynb
```

### To train all models from scratch:

Run all cells top to bottom in order.

### To load saved checkpoints (no retraining):

Run cells 1 to 22 to define everything then run the
checkpoint loading cell directly. Takes about 10 minutes.

### To reproduce exact results:

```
1. Download MARIDA dataset
2. Place in MARIDA/ folder as shown above
3. Run cells 1-22
4. Run checkpoint evaluation cell
5. Results match Table III in report
```

---

## Hardware

Tested on:

```
GPU    : NVIDIA GeForce RTX 5060 Laptop GPU 8GB VRAM
CPU    : AMD Ryzen 9HX
RAM    : 16GB
OS     : Ubuntu 24
Python : 3.13
PyTorch: 2.12.0
CUDA   : 12.8
```

Minimum requirements:

```
GPU    : 6GB VRAM
RAM    : 16GB
Storage: 10GB free (dataset + checkpoints)
```

---

## Known Issues

```
1. Single worker DataLoader required
   rasterio is not fork-safe with multiple workers

2. Minimum 6GB GPU VRAM needed for training
   DeepSpectralUNet with batch size 4

3. Training may be interrupted on low VRAM GPUs
   Checkpoint is saved automatically at best val F1
   Resume from checkpoint without retraining
```

---

