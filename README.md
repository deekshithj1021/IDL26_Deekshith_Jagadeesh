# IDL26 — Final Assignment: Operation Cyber-Histology
**Post-Incident Pipeline Reconstruction and ML Engineering**

---

## Author
| Name | Enrollment Number |
|------|------------------|
| Deekshith Jagadeesh | [10000759] |

---

## Overview
This repository contains the fully restored, debugged, and optimized 
machine learning pipeline for BioHealth Diagnostics Global's 
multi-class clinical image triage system.

The pipeline supports three model architectures (AlexNet, VGG16, ResNet18)
across four medical imaging datasets (cells, chest, lesions, orgs),
with additional transfer learning support for the scarce organs dataset.

---

## Repository Structure


IDL26_Deekshith_Jagadeesh/
├── Code/
│   ├── data.py          # Data loading and train/val/test splitting
│   ├── fit.py           # Training loop and evaluation logic
│   ├── models.py        # AlexNet, VGG16, ResNet18 architectures
│   ├── train.py         # Single config training entry point
│   ├── run_all.py       # Automated runner for all combinations
│   └── config.json      # Configuration file for single runs
├── AUDIT_LOG.md         # Full bug audit documentation
├── REPORT.md            # Benchmark results and analysis
└── README.md            # This file

---

## Datasets
Data files are not included in this repository due to file size.
Download from: https://cloud.fiw.fhws.de/s/LpYa2dCW85kwdNn

Place downloaded `.pt` files in a `Data/` folder at the project root:

Data/
├── cells_data.pt      # 13,671 train | 3,421 test | 8 classes  | RGB
├── chest_data.pt      # 5,232 train  | 624 test   | 2 classes  | Grayscale
├── lesions_data.pt    # 8,010 train  | 2,005 test | 7 classes  | RGB
├── orgs_data.pt       # 15,367 train | 8,216 test | 11 classes | Grayscale
└── organs_data.pt     # 500 train    | 200 test   | 11 classes | Grayscale



---

## Prerequisites

- Python 3.9+
- PyTorch 2.0+

Install all dependencies:
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
and prints a final summary table.

---

## Configuration Reference

| Parameter | Description | Example Values |
|-----------|-------------|---------------|
| MODEL | Architecture to use | ResNet18, VGG16, AlexNet |
| DATA | Dataset name | cells, chest, lesions, orgs |
| DATA_PATH | Path to Data folder | ../Data |
| CHANNELS | Input image channels | 3 (RGB), 1 (Grayscale) |
| NUM_CLASSES | Number of output classes | 8, 2, 7, 11 |
| BATCH_SIZE | Images per training batch | 32 |
| EPOCHS | Number of training epochs | 10 |
| LEARNING_RATE | Adam optimizer step size | 0.001 |
| DROP_RATE | Dropout probability | 0.5 |

---

## Dataset Configuration Quick Reference

| Dataset | CHANNELS | NUM_CLASSES |
|---------|----------|-------------|
| cells | 3 | 8 |
| chest | 1 | 2 |
| lesions | 3 | 7 |
| orgs | 1 | 11 |
| organs | 1 | 11 |

---

## Bugs Fixed
Eight critical bugs were identified and fixed in the recovered codebase.
Full details are documented in `AUDIT_LOG.md`.

| # | File | Bug |
|---|------|-----|
| 1 | fit.py | Missing zero_grad() causing gradient explosion |
| 2 | data.py | Validation data leaking into training set |
| 3 | data.py | Wrong label shape causing CrossEntropyLoss crash |
| 4 | models.py | Identity activation killing network learning |
| 5 | models.py | ResNet18 forward() returning None |
| 6 | models.py | Wrong padding on 1x1 convolution |
| 7 | models.py | AlexNet ignoring in_channels and num_classes |
| 8 | train.py | Hardcoded drop_rate=0.99 and missing config.json |


