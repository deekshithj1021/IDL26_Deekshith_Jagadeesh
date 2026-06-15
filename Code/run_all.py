"""
MAI/IDL SS26 - Final Assignment
Automated training runner for all model/dataset combinations.
Includes profiling for green initiative analysis.
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
import models
from data import get_loaders
from fit import Trainer

# ── All configurations to run ───────────────────────────────────────────────
CONFIGS = [
    {"MODEL": "ResNet18", "DATA": "cells",   "CHANNELS": 3, "NUM_CLASSES": 8,  "EPOCHS": 30},
    {"MODEL": "ResNet18", "DATA": "chest",   "CHANNELS": 1, "NUM_CLASSES": 2,  "EPOCHS": 30},
    {"MODEL": "ResNet18", "DATA": "lesions", "CHANNELS": 3, "NUM_CLASSES": 7,  "EPOCHS": 30},
    {"MODEL": "ResNet18", "DATA": "orgs",    "CHANNELS": 1, "NUM_CLASSES": 11, "EPOCHS": 30},
    {"MODEL": "VGG16",    "DATA": "cells",   "CHANNELS": 3, "NUM_CLASSES": 8,  "EPOCHS": 30},
    {"MODEL": "VGG16",    "DATA": "chest",   "CHANNELS": 1, "NUM_CLASSES": 2,  "EPOCHS": 30},
    {"MODEL": "VGG16",    "DATA": "lesions", "CHANNELS": 3, "NUM_CLASSES": 7,  "EPOCHS": 30},
    {"MODEL": "VGG16",    "DATA": "orgs",    "CHANNELS": 1, "NUM_CLASSES": 11, "EPOCHS": 30},
    {"MODEL": "AlexNet",  "DATA": "cells",   "CHANNELS": 3, "NUM_CLASSES": 8,  "EPOCHS": 30},
    {"MODEL": "AlexNet",  "DATA": "chest",   "CHANNELS": 1, "NUM_CLASSES": 2,  "EPOCHS": 30},
    {"MODEL": "AlexNet",  "DATA": "lesions", "CHANNELS": 3, "NUM_CLASSES": 7,  "EPOCHS": 30},
    {"MODEL": "AlexNet",  "DATA": "orgs",    "CHANNELS": 1, "NUM_CLASSES": 11, "EPOCHS": 30},
    {"MODEL": "MiniNet",  "DATA": "cells",   "CHANNELS": 3, "NUM_CLASSES": 8,  "EPOCHS": 30},
    {"MODEL": "MiniNet",  "DATA": "chest",   "CHANNELS": 1, "NUM_CLASSES": 2,  "EPOCHS": 30},
    {"MODEL": "MiniNet",  "DATA": "lesions", "CHANNELS": 3, "NUM_CLASSES": 7,  "EPOCHS": 30},
    {"MODEL": "MiniNet",  "DATA": "orgs",    "CHANNELS": 1, "NUM_CLASSES": 11, "EPOCHS": 30},
]

# ── Shared settings ─────────────────────────────────────────────────────────
DATA_PATH     = "../Data"
BATCH_SIZE    = 32
LEARNING_RATE = 0.001
DROP_RATE     = 0.5

# ── Device setup ────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Running on device: {device}\n")

# ── Helper: count parameters ────────────────────────────────────────────────
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ── Helper: measure inference latency ───────────────────────────────────────
def measure_latency(model, channels, device, n=100):
    model.eval()
    dummy = torch.randn(1, channels, 64, 64).to(device)
    with torch.no_grad():
        start = time.time()
        for _ in range(n):
            model(dummy)
        elapsed = (time.time() - start) / n * 1000
    return elapsed

# ── Results storage ─────────────────────────────────────────────────────────
results = []

# ── Main loop ───────────────────────────────────────────────────────────────
for cfg in CONFIGS:
    print(f"\n{'='*60}")
    print(f"  Model: {cfg['MODEL']}  |  Dataset: {cfg['DATA']}")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader = get_loaders(
        data=cfg["DATA"],
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE
    )

    model_class = getattr(models, cfg["MODEL"])
    model = model_class(
        in_channels=cfg["CHANNELS"],
        num_classes=cfg["NUM_CLASSES"],
        drop_rate=DROP_RATE
    ).to(device)

    num_params = count_params(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    trainer = Trainer(model, criterion, optimizer, device, patience=5)

    # Training with timing
    train_start = time.time()
    trainer.fit(train_loader, val_loader, epochs=cfg["EPOCHS"])
    train_time = time.time() - train_start

    # Test accuracy
    test_loss, test_acc = trainer.evaluate(test_loader)

    # Inference latency
    latency = measure_latency(model, cfg["CHANNELS"], device)

    print(f">>> Test Accuracy:      {test_acc:.2f}%")
    print(f">>> Parameters:         {num_params:,}")
    print(f">>> Training Time:      {train_time:.1f}s")
    print(f">>> Inference Latency:  {latency:.2f}ms/sample")

    results.append({
        "model":    cfg["MODEL"],
        "dataset":  cfg["DATA"],
        "test_acc": round(test_acc, 2),
        "params":   num_params,
        "train_time": round(train_time, 1),
        "latency":  round(latency, 2)
    })

# ── Final summary table ──────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"  FINAL RESULTS SUMMARY")
print(f"{'='*80}")
print(f"{'Model':<12} {'Dataset':<10} {'Test Acc':>10} {'Params':>12} {'Train Time':>12} {'Latency':>10}")
print(f"{'-'*68}")
for r in results:
    print(f"{r['model']:<12} {r['dataset']:<10} {r['test_acc']:>9.2f}% "
          f"{r['params']:>12,} {r['train_time']:>11.1f}s {r['latency']:>9.2f}ms")
print(f"{'='*80}")