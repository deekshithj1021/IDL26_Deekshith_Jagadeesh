"""
MAI/IDL SS26 - Final Assignment
Automated training runner for all model/dataset combinations.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import models
from data import get_loaders
from fit import Trainer

# ── All configurations to run ──────────────────────────────────────────────
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
]

# ── Shared settings ─────────────────────────────────────────────────────────
DATA_PATH    = "../Data"
BATCH_SIZE   = 32
LEARNING_RATE = 0.001
DROP_RATE    = 0.5

# ── Device setup ────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Running on device: {device}\n")

# ── Results storage ─────────────────────────────────────────────────────────
results = []

# ── Main loop ───────────────────────────────────────────────────────────────
for cfg in CONFIGS:
    print(f"\n{'='*60}")
    print(f"  Model: {cfg['MODEL']}  |  Dataset: {cfg['DATA']}")
    print(f"{'='*60}")

    # Load data
    train_loader, val_loader, test_loader = get_loaders(
        data=cfg["DATA"],
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE
    )

    # Build model
    model_class = getattr(models, cfg["MODEL"])
    model = model_class(
        in_channels=cfg["CHANNELS"],
        num_classes=cfg["NUM_CLASSES"],
        drop_rate=DROP_RATE
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train
    trainer = Trainer(model, criterion, optimizer, device, patience=5)
    trainer.fit(train_loader, val_loader, epochs=cfg["EPOCHS"])

    # Evaluate on test set
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f">>> Test Accuracy: {test_acc:.2f}%")

    # Save result
    results.append({
        "model":    cfg["MODEL"],
        "dataset":  cfg["DATA"],
        "test_acc": round(test_acc, 2)
    })

# ── Print final summary table ────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  FINAL RESULTS SUMMARY")
print(f"{'='*60}")
print(f"{'Model':<12} {'Dataset':<10} {'Test Acc':>10}")
print(f"{'-'*35}")
for r in results:
    print(f"{r['model']:<12} {r['dataset']:<10} {r['test_acc']:>9.2f}%")
print(f"{'='*60}")