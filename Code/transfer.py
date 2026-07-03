"""
MAI/IDL SS26 - Final Assignment
Transfer Learning on scarce organs dataset.
Compares training from scratch vs fine-tuning from orgs-pretrained weights.
Addresses Task 3: Knowledge Transfer Adaptation for limited training data.
"""
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import models
from data import get_loaders
from fit import Trainer


# ── Load shared settings from config ────────────────────────────────────────
with open("config.json", "r") as f:
    config = json.load(f)

shared = config["SHARED"]

DATA_PATH     = shared["DATA_PATH"]
LEARNING_RATE = shared["LEARNING_RATE"]
DROP_RATE     = shared["DROP_RATE"]

# organs specific settings — fixed because this script is only for organs
BATCH_SIZE  = 16    # smaller than shared because organs dataset is tiny (500 samples)
EPOCHS      = 30    # max epochs for both approaches
CHANNELS    = 1     # organs images are grayscale
NUM_CLASSES = 11    # 11 organ types


# ── Device setup ────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Running on device: {device}\n")


# ── Load organs data (500 samples) ──────────────────────────────────────────
print("Loading organs dataset (scarce — 500 samples)...")
train_loader, val_loader, test_loader = get_loaders(
    data="organs",
    data_path=DATA_PATH,
    batch_size=BATCH_SIZE
)


# ════════════════════════════════════════════════════════════════════════════
# APPROACH 1: Train from scratch on organs
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  APPROACH 1: Training from scratch on organs")
print("="*60)

model_scratch = models.ResNet18(
    in_channels=CHANNELS,
    num_classes=NUM_CLASSES,
    drop_rate=DROP_RATE
).to(device)

optimizer_scratch = optim.Adam(
    model_scratch.parameters(),
    lr=LEARNING_RATE
)
trainer_scratch = Trainer(
    model_scratch,
    nn.CrossEntropyLoss(),
    optimizer_scratch,
    device
)

start = time.time()
trainer_scratch.fit(train_loader, val_loader, epochs=EPOCHS)
scratch_time = time.time() - start

_, scratch_test_acc = trainer_scratch.evaluate(test_loader)
print(f">>> Scratch Test Accuracy: {scratch_test_acc:.2f}%")
print(f">>> Training Time:         {scratch_time:.1f}s")


# ════════════════════════════════════════════════════════════════════════════
# APPROACH 2: Pre-train on orgs, fine-tune on organs
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  APPROACH 2: Pre-train on orgs, fine-tune on organs")
print("="*60)

# ── Step 1: Pre-train on orgs (large dataset, same domain) ──────────────────
print("\nStep 1: Pre-training on orgs dataset (15,367 samples)...")

# orgs test loader discarded — we only need train/val for pre-training
orgs_train, orgs_val, _ = get_loaders(
    data="orgs",
    data_path=DATA_PATH,
    batch_size=shared["BATCH_SIZE"]   # use standard batch size for large dataset
)

model_pretrained = models.ResNet18(
    in_channels=CHANNELS,
    num_classes=NUM_CLASSES,
    drop_rate=DROP_RATE
).to(device)

optimizer_pre = optim.Adam(
    model_pretrained.parameters(),
    lr=LEARNING_RATE
)
trainer_pre = Trainer(
    model_pretrained,
    nn.CrossEntropyLoss(),
    optimizer_pre,
    device
)
trainer_pre.fit(orgs_train, orgs_val, epochs=15)


# ── Step 2: Fine-tune on organs (small dataset) ──────────────────────────────
print("\nStep 2: Fine-tuning on organs dataset (500 samples)...")

# Unfreeze all layers so the whole network can adapt
for param in model_pretrained.parameters():
    param.requires_grad = True

# Use very small learning rate to preserve pre-trained knowledge
# 20x smaller than normal to prevent catastrophic forgetting
optimizer_ft = optim.Adam(
    model_pretrained.parameters(),
    lr=0.00005
)
trainer_ft = Trainer(
    model_pretrained,
    nn.CrossEntropyLoss(),
    optimizer_ft,
    device
)

start = time.time()
trainer_ft.fit(train_loader, val_loader, epochs=EPOCHS)
ft_time = time.time() - start

_, ft_test_acc = trainer_ft.evaluate(test_loader)
print(f">>> Fine-tuned Test Accuracy: {ft_test_acc:.2f}%")
print(f">>> Fine-tuning Time:         {ft_time:.1f}s")


# ════════════════════════════════════════════════════════════════════════════
# RESULTS COMPARISON
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  TRANSFER LEARNING RESULTS SUMMARY")
print("="*60)
print(f"{'Approach':<30} {'Test Acc':>10} {'Time':>10}")
print("-"*52)
print(f"{'Scratch (organs only)':<30} "
      f"{scratch_test_acc:>9.2f}% {scratch_time:>9.1f}s")
print(f"{'Transfer (orgs → organs)':<30} "
      f"{ft_test_acc:>9.2f}% {ft_time:>9.1f}s")
print("="*60)
print(f"\nImprovement from transfer learning: "
      f"{ft_test_acc - scratch_test_acc:+.2f}%")
print(f"Required minimum accuracy:          40%")
print(f"Best achieved:                      "
      f"{max(scratch_test_acc, ft_test_acc):.2f}%")