# IDL26 — Final Assignment: Operation Cyber-Histology
**Post-Incident Pipeline Reconstruction and ML Engineering**

---

## Authors
| Name | Enrollment Number |
|------|------------------|
| Deekshith Jagadeesh | [10000759] |
| [Prahas Hegde] | [Teammate Enrollment Number] |

---

## Overview
This repository contains the fully restored, debugged, and optimized 
machine learning pipeline for BioHealth Diagnostics Global's 
multi-class clinical image triage system. The original codebase was 
recovered after a malicious environment-wipe and has been fully 
reconstructed, audited, and extended.

The pipeline supports three model architectures:
- AlexNet (Krizhevsky et al., 2012)
- VGG16 in Configuration C (Simonyan & Zisserman, 2014)
- ResNet18 (He et al., 2016)

Across four medical imaging datasets:
- cells — Blood cell microscopy (8 classes, RGB)
- chest — Chest X-rays (2 classes, Grayscale)
- lesions — Skin lesion images (7 classes, RGB)
- orgs — Organ images (11 classes, Grayscale)

With additional transfer learning support for the scarce organs dataset.

---

## Repository Structure

```
IDL26_Deekshith_Jagadeesh/
├── Code/
│   ├── data.py             # Data loading and train/val/test splitting
│   ├── fit.py              # Training loop and evaluation logic
│   ├── models.py           # AlexNet, VGG16, ResNet18 architectures
│   ├── train.py            # Single config training entry point
│   ├── run_all.py          # Automated runner for all combinations
│   └── config.json         # Configuration file for single runs
├── AUDIT_LOG.md            # Full bug audit documentation
├── REPORT.md               # Benchmark results and analysis
└── README.md               # This file
```

---

## Datasets

Data files are not included in this repository due to file size.
Download from: https://cloud.fiw.fhws.de/s/LpYa2dCW85kwdNn

Place downloaded `.pt` files in a `Data/` folder at the project root:

```
Data/
├── cells_data.pt       # 13,671 train | 3,421 test | 8 classes  | RGB
├── chest_data.pt       # 5,232 train  | 624 test   | 2 classes  | Grayscale
├── lesions_data.pt     # 8,010 train  | 2,005 test | 7 classes  | RGB
├── orgs_data.pt        # 15,367 train | 8,216 test | 11 classes | Grayscale
└── organs_data.pt      # 500 train    | 200 test   | 11 classes | Grayscale
```

---

## Prerequisites

- Python 3.9+
- PyTorch 2.0+
- torchvision
- numpy
- matplotlib
- scikit-learn

Install all dependencies with one command:

```bash
pip3 install torch torchvision numpy matplotlib scikit-learn
```

---

## How to Run

### Option 1 — Train a Single Model

Edit `Code/config.json` to set your desired configuration:

```json
{
    "MODEL": "ResNet18",
    "DATA": "cells",
    "DATA_PATH": "../Data",
    "CHANNELS": 3,
    "NUM_CLASSES": 8,
    "BATCH_SIZE": 32,
    "EPOCHS": 10,
    "LEARNING_RATE": 0.001,
    "DROP_RATE": 0.5
}
```

Then run:

```bash
cd Code
python3 train.py
```

### Option 2 — Train All Combinations Automatically

```bash
cd Code
python3 run_all.py
```

This automatically trains all 12 model/dataset combinations
and prints a full summary table at the end.

---

## Configuration Reference

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| MODEL | Architecture to use | ResNet18, VGG16, AlexNet |
| DATA | Dataset name | cells, chest, lesions, orgs |
| DATA_PATH | Path to Data folder | ../Data |
| CHANNELS | Input image channels | 3 (RGB), 1 (Grayscale) |
| NUM_CLASSES | Number of output classes | 8, 2, 7, 11 |
| BATCH_SIZE | Images per training batch | 32 |
| EPOCHS | Number of training epochs | 10 |
| LEARNING_RATE | Adam optimizer learning rate | 0.001 |
| DROP_RATE | Dropout probability | 0.5 |

---

## Dataset Configuration Quick Reference

| Dataset | CHANNELS | NUM_CLASSES | Size |
|---------|----------|-------------|------|
| cells | 3 | 8 | Large (13,671 samples) |
| chest | 1 | 2 | Medium (5,232 samples) |
| lesions | 3 | 7 | Medium (8,010 samples) |
| orgs | 1 | 11 | Large (15,367 samples) |
| organs | 1 | 11 | Scarce (500 samples) |

---

## Bugs Fixed

Eight critical bugs were identified and fixed in the recovered codebase.
Full details with root cause analysis and commit hashes are in `AUDIT_LOG.md`.

| # | File | Bug Type | Description |
|---|------|----------|-------------|
| 1 | fit.py | Gradient Failure | Missing zero_grad() causing gradient explosion |
| 2 | data.py | Silent Logical Flaw | Validation data leaking into training set |
| 3 | data.py | Runtime Crash | Wrong label shape causing CrossEntropyLoss crash |
| 4 | models.py | Numerical Failure | Identity activation killing all network learning |
| 5 | models.py | Silent Bug | ResNet18 forward() returning None silently |
| 6 | models.py | Runtime Crash | Wrong padding on 1x1 convolution in VGGBlock |
| 7 | models.py | Runtime Crash | AlexNet ignoring in_channels and num_classes |
| 8 | train.py | Rigid Infrastructure | Hardcoded drop_rate=0.99, missing config.json |

---

## Architecture Notes

### ResNet18
- Adapted for 64x64 inputs with reduced stride in the first conv layer
- Flexible activation function via global `activation_str` variable
- Uses skip connections to prevent vanishing gradients in deep layers
- AdaptiveAvgPool2d before classifier for input size flexibility

### VGG16
- Configuration C from Simonyan & Zisserman (2014)
- Uses 1x1 convolutions in the third layer of 3-conv blocks
- Fixed padding logic: 1x1 convolutions use padding=0, 3x3 use padding=1
- Adapted classifier head for 64x64 input size

### AlexNet
- Adapted for 64x64 inputs with reduced kernel sizes and strides
- Fully configurable input channels and output classes
- Dropout regularization with configurable drop rate
- Batch normalization after first two conv layers

---

## Device Support

The pipeline automatically detects and uses the best available device:

```
Apple Silicon Mac (MPS)  → automatically detected and used
NVIDIA GPU (CUDA)        → automatically detected and used
CPU                      → fallback, works everywhere but slower
```

No manual configuration needed — the device is selected automatically
at runtime in both `train.py` and `run_all.py`.

---

## Performance Targets

| Dataset | Required Accuracy | 
|---------|------------------|
| cells   | 90%              |
| chest   | 87%              |
| lesions | 67%              |
| orgs    | 83%              |

Full benchmark results for all model/dataset combinations
are documented in `REPORT.md`.

---

## Submission
- **Platform:** e-learning platform
- **Format:** Direct link to this repository
- **Branch:** main
- **Deadline:** 09.07.2026, 23:59 German Time