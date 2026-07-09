# IDL26 — Operation Cyber-Histology
**Post-Incident Pipeline Reconstruction and ML Engineering**

**Authors:** 
| Name | Enrollment Number |
|------|------------------|
| Deekshith Jagadeesh | 10000759 |
| Prahas Hegde | 10001212 |

---

**Course:** MAI/IDL SS26 — Final Assignment

**Date:** 09 July 2026

---

## Overview

This repository contains our restored and extended machine
learning pipeline for BioHealth Diagnostics Global's
clinical image classification system. The original codebase
was recovered after deliberate sabotage and contained
multiple bugs across four files. We fixed all bugs, rebuilt
the missing configuration and testing infrastructure, added
a lightweight green model called PlantNet, and implemented
transfer learning for the scarce organs dataset.

The pipeline classifies medical images across four datasets, cells, chest, lesions and orgs, using three restored
architectures (AlexNet, VGG16, ResNet18) and one new
architecture we designed (PlantNet).

---

## Repository Structure

```
IDL26_Deekshith_Jagadeesh/
│
├── Code/
│   ├── data.py          # loads data, splits train/val/test
│   ├── fit.py           # training loop with best weight saving
│   ├── models.py        # AlexNet, VGG16, ResNet18, PlantNet
│   ├── train.py         # runs all combinations from config
│   ├── benchmark.py     # full evaluation with all metrics
│   ├── transfer.py      # transfer learning for organs dataset
│   └── config.json      # all training configurations
│
├── Data/                # not included — download separately
│   ├── cells_data.pt
│   ├── chest_data.pt
│   ├── lesions_data.pt
│   ├── orgs_data.pt
│   └── organs_data.pt
│
├── AUDIT_LOG.md         # all bugs found and fixed
├── REPORT.md            # benchmark results and analysis
└── README.md            # this file
```

---

## Datasets

Data files are not included in this repository because
they are too large for GitHub. Download them from:

```
https://cloud.fiw.fhws.de/s/LpYa2dCW85kwdNn
```

After downloading create a Data folder at the project
root and place all .pt files inside it:

```
DL_FINAL_ASIGN/
├── Code/
└── Data/
    ├── cells_data.pt
    ├── chest_data.pt
    ├── lesions_data.pt
    ├── orgs_data.pt
    └── organs_data.pt
```

Each dataset has different properties:

| Dataset | Channels | Classes | Train Samples | Test Samples |
|---------|----------|---------|---------------|--------------|
| cells | 3 (RGB) | 8 | 13,671 | 3,421 |
| chest | 1 (grayscale) | 2 | 5,232 | 624 |
| lesions | 3 (RGB) | 7 | 8,010 | 2,005 |
| orgs | 1 (grayscale) | 11 | 15,367 | 8,216 |
| organs | 1 (grayscale) | 11 | 500 | 200 |

---

## Prerequisites

```
Python 3.9 or higher
PyTorch 2.0 or higher
```

We developed and tested this on a Mac with Apple Silicon
(M-series chip) using the MPS GPU backend. The code
automatically detects MPS, CUDA or CPU in that order.

---

## Installation

Install all required dependencies with one command:

```bash
pip install torch torchvision scikit-learn numpy
```

If you are on Mac and pip points to the wrong Python
version use the full path:

```bash
/Library/Developer/CommandLineTools/usr/bin/python3.9 -m pip install torch torchvision scikit-learn numpy
```

---

## How To Run

All scripts must be run from inside the Code folder:

```bash
cd Code
```

---

### Option 1 — Train All Models (train.py)

Reads config.json and trains all model/dataset combinations
one by one. Prints test accuracy after each run.

```bash
python3 train.py
```

---

### Option 2 — Full Benchmark (benchmark.py)

Runs all combinations and calculates accuracy, precision,
recall, F1-score, training time and inference latency.
Prints a complete results table at the end.

```bash
python3 benchmark.py
```

This is the main evaluation script. Running all 16
combinations takes approximately 1-2 hours on Mac MPS.

---

### Option 3 — Transfer Learning (transfer.py)

Runs transfer learning experiments on the scarce organs
dataset. Compares scratch training vs pre-training on
orgs then fine-tuning on organs.

```bash
python3 transfer.py
```

---

## Configuration

All training settings are controlled from config.json.
No need to edit any Python file to change models or
datasets.

The config has two sections:

**SHARED** — settings that apply to all runs:
```json
{
    "SHARED": {
        "DATA_PATH": "../Data",
        "BATCH_SIZE": 64,
        "LEARNING_RATE": 0.001,
        "DROP_RATE": 0.5
    }
}
```

**CONFIGS** — list of all model/dataset combinations:
```json
{
    "CONFIGS": [
        {"MODEL": "ResNet18", "DATA": "cells",
         "CHANNELS": 3, "NUM_CLASSES": 8, "EPOCHS": 10},
        {"MODEL": "VGG16", "DATA": "chest",
         "CHANNELS": 1, "NUM_CLASSES": 2, "EPOCHS": 10}
    ]
}
```

To add a new combination just add a new entry to CONFIGS.
To change epochs for a specific run change the EPOCHS
value in that entry only.

Available models: ResNet18, VGG16, AlexNet, PlantNet

Dataset channel and class reference:

| Dataset | CHANNELS | NUM_CLASSES |
|---------|----------|-------------|
| cells | 3 | 8 |
| chest | 1 | 2 |
| lesions | 3 | 7 |
| orgs | 1 | 11 |
| organs | 1 | 11 |

---

## What We Fixed

The recovered codebase had 15 bugs across 4 files.
Full details are in AUDIT_LOG.md. Brief summary:

| File | Bugs Fixed |
|------|-----------|
| fit.py | Missing zero_grad, variable shadowing |
| data.py | Data leak, wrong label shapes |
| models.py | Identity activation, missing return, wrong padding, channel update, AlexNet params, classifier size |
| train.py | Missing config, no MPS support, test loader discarded, drop_rate 0.99 |

---

## Authors

| Name | Role |
|------|------|
| Deekshith Jagadeesh | Task 1 (bug fixes, benchmark) Task 2 (PlantNet) |
| Prahas Hegde | Task 3 (transfer learning) |