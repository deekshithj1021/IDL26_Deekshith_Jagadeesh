"""
MAI/IDL SS26 - Final Assignment
Transfer Learning on scarce organs dataset.

ResNet18 selected for final run — highest absolute
accuracy (64%) and most robust across both approaches.
"""
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import models
from data import get_loaders
from fit import Trainer


with open("config.json", "r") as f:
    config = json.load(f)

shared = config["SHARED"]

DATA_PATH     = shared["DATA_PATH"]
LEARNING_RATE = shared["LEARNING_RATE"]
DROP_RATE     = shared["DROP_RATE"]

BATCH_SIZE      = 16       # smaller than shared — organs has only 500 samples
EPOCHS          = 30       # final run — more epochs than exploration
PRETRAIN_EPOCHS = 20       # pre-train on orgs for more epochs
FINE_TUNE_LR    = 0.00005  # 20x smaller than normal — prevents forgetting
CHANNELS        = 1        # organs is grayscale
NUM_CLASSES     = 11       # 11 organ types


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Running on device: {device}\n")
print("Selected model: ResNet18")
print("Reason: highest absolute accuracy in exploration (64%)")
print("        robust on both scratch and transfer approaches\n")

# Load organs data
print("Loading organs dataset (scarce — 500 samples)...")
train_loader, val_loader, test_loader = get_loaders(data="organs",data_path=DATA_PATH,batch_size=BATCH_SIZE)

# APPROACH 1: Train ResNet18 from scratch on organs (baseline)
print("\n" + "="*60)
print("  APPROACH 1: ResNet18 from scratch on organs")
print("="*60)

model_scratch = models.ResNet18(in_channels=CHANNELS,num_classes=NUM_CLASSES,drop_rate=DROP_RATE).to(device)

optimizer_scratch = optim.Adam(model_scratch.parameters(), lr=LEARNING_RATE)
trainer_scratch = Trainer(model_scratch, nn.CrossEntropyLoss(), optimizer_scratch, device)

start = time.time()
trainer_scratch.fit(train_loader, val_loader, epochs=EPOCHS)
scratch_time = round(time.time() - start, 1)

_, scratch_acc = trainer_scratch.evaluate(test_loader)
print(f">>> Scratch Accuracy: {scratch_acc:.2f}%")
print(f">>> Training Time:    {scratch_time:.1f}s")

# APPROACH 2: Pre-train ResNet18 on orgs then fine-tune on organs
print("\n" + "="*60)
print("  APPROACH 2: ResNet18 transfer from orgs to organs")
print("="*60)

# Step 1: Pre-train on orgs (large dataset same domain)
print(f"\nStep 1: Pre-training on orgs ({PRETRAIN_EPOCHS} epochs)...")

orgs_train, orgs_val, _ = get_loaders(data="orgs",data_path=DATA_PATH,batch_size=shared["BATCH_SIZE"])

model_ft = models.ResNet18(in_channels=CHANNELS,num_classes=NUM_CLASSES,drop_rate=DROP_RATE).to(device)

optimizer_pre = optim.Adam(model_ft.parameters(), lr=LEARNING_RATE)
trainer_pre = Trainer(model_ft, nn.CrossEntropyLoss(), optimizer_pre, device)
trainer_pre.fit(orgs_train, orgs_val, epochs=PRETRAIN_EPOCHS)

# step 2 — fine-tune on organs with tiny learning rate
# unfreeze all layers so whole network adapts to organs distribution
print(f"\nStep 2: Fine-tuning on organs ({EPOCHS} epochs)...")
print(f"        Using lr={FINE_TUNE_LR} to prevent catastrophic forgetting")

for param in model_ft.parameters():
    param.requires_grad = True

# tiny lr preserves pre-trained knowledge while adapting to organs
optimizer_ft = optim.Adam(model_ft.parameters(), lr=FINE_TUNE_LR)
trainer_ft = Trainer(model_ft, nn.CrossEntropyLoss(), optimizer_ft, device)

start = time.time()
trainer_ft.fit(train_loader, val_loader, epochs=EPOCHS)
ft_time = round(time.time() - start, 1)

_, ft_acc = trainer_ft.evaluate(test_loader)
print(f">>> Fine-tuned Accuracy: {ft_acc:.2f}%")
print(f">>> Fine-tuning Time:    {ft_time:.1f}s")


print("\n" + "="*60)
print("  FINAL TRANSFER LEARNING RESULTS — ResNet18")
print("="*60)
print(f"{'Approach':<30} {'Accuracy':>10} {'Time':>10}")
print("-"*52)
print(f"{'Scratch (organs only)':<30} {scratch_acc:>9.2f}% {scratch_time:>9.1f}s")
print(f"{'Transfer (orgs → organs)':<30} {ft_acc:>9.2f}% {ft_time:>9.1f}s")
print("="*60)
print(f"\nImprovement from transfer learning: {ft_acc - scratch_acc:+.2f}%")
print(f"Required minimum accuracy: 40%")
print(f"Best achieved: {max(scratch_acc, ft_acc):.2f}%")