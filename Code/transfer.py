"""
MAI/IDL SS26 - Final Assignment
Transfer Learning on scarce organs dataset.
Exploration run: tests all 4 architectures with scratch vs fine-tuning.
Selects best model for final production run.
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

DATA_PATH     = shared["DATA_PATH"]    # from config — shared with pipeline
LEARNING_RATE = shared["LEARNING_RATE"] # from config — shared with pipeline
DROP_RATE     = shared["DROP_RATE"]    # from config — shared with pipeline

# organs specific — unique to this script
BATCH_SIZE      = 16      # smaller than shared — organs has only 500 samples
EPOCHS          = 15      # exploration run — quick results
PRETRAIN_EPOCHS = 10      # orgs pre-training epochs
FINE_TUNE_LR    = 0.00005 # 20x smaller than normal — prevents forgetting
CHANNELS        = 1       # organs is grayscale
NUM_CLASSES     = 11      # 11 organ types


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Running on device: {device}\n")


print("Loading organs dataset (scarce — 500 samples)...")
train_loader, val_loader, test_loader = get_loaders(
    data="organs",
    data_path=DATA_PATH,
    batch_size=BATCH_SIZE
)

# Models to test 
MODEL_NAMES = ["ResNet18", "VGG16", "AlexNet", "PlantNet"]

results = []

for model_name in MODEL_NAMES:
    print(f"\n{'='*60}")
    print(f"  Testing: {model_name}")
    print(f"{'='*60}")

    # APPROACH 1: Train from scratch 
    print(f"\n--- Approach 1: {model_name} from scratch ---")

    model_scratch = getattr(models, model_name)(
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
    scratch_time = round(time.time() - start, 1)

    _, scratch_acc = trainer_scratch.evaluate(test_loader)
    print(f">>> Scratch Accuracy: {scratch_acc:.2f}%")
    print(f">>> Training Time:    {scratch_time:.1f}s")

    # APPROACH 2: Pre-train on orgs then fine-tune on organs 
    print(f"\n--- Approach 2: {model_name} transfer from orgs ---")

    # Step 1 — Pre-train on orgs (large dataset same domain)
    print(f"Step 1: Pre-training on orgs ({PRETRAIN_EPOCHS} epochs)...")
    orgs_train, orgs_val, _ = get_loaders(
        data="orgs",
        data_path=DATA_PATH,
        batch_size=shared["BATCH_SIZE"]
    )

    model_ft = getattr(models, model_name)(
        in_channels=CHANNELS,
        num_classes=NUM_CLASSES,
        drop_rate=DROP_RATE
    ).to(device)

    optimizer_pre = optim.Adam(
        model_ft.parameters(),
        lr=LEARNING_RATE
    )
    trainer_pre = Trainer(
        model_ft,
        nn.CrossEntropyLoss(),
        optimizer_pre,
        device
    )
    trainer_pre.fit(orgs_train, orgs_val, epochs=PRETRAIN_EPOCHS)

    # Step 2 — Fine-tune on organs (small dataset)
    print(f"Step 2: Fine-tuning on organs ({EPOCHS} epochs)...")

    for param in model_ft.parameters():
        param.requires_grad = True

    optimizer_ft = optim.Adam(
        model_ft.parameters(),
        lr=FINE_TUNE_LR
    )
    trainer_ft = Trainer(
        model_ft,
        nn.CrossEntropyLoss(),
        optimizer_ft,
        device
    )

    start = time.time()
    trainer_ft.fit(train_loader, val_loader, epochs=EPOCHS)
    ft_time = round(time.time() - start, 1)

    _, ft_acc = trainer_ft.evaluate(test_loader)
    print(f">>> Fine-tuned Accuracy: {ft_acc:.2f}%")
    print(f">>> Fine-tuning Time:    {ft_time:.1f}s")

    results.append({
        "model":        model_name,
        "scratch_acc":  round(scratch_acc, 2),
        "ft_acc":       round(ft_acc, 2),
        "improvement":  round(ft_acc - scratch_acc, 2),
        "scratch_time": scratch_time,
        "ft_time":      ft_time
    })


print(f"\n{'='*70}")
print(f"  TRANSFER LEARNING EXPLORATION RESULTS")
print(f"{'='*70}")
print(f"{'Model':<12} {'Scratch':>10} {'Transfer':>10} {'Gain':>8} {'Time':>8}")
print(f"{'-'*52}")
for r in results:
    print(f"{r['model']:<12} {r['scratch_acc']:>9.2f}% "
          f"{r['ft_acc']:>9.2f}% {r['improvement']:>+7.2f}% "
          f"{r['ft_time']:>7.1f}s")
print(f"{'='*70}")

best = max(results, key=lambda x: x['ft_acc'])
print(f"\nBest model for organs: {best['model']}")
print(f"Best transfer accuracy: {best['ft_acc']:.2f}%")
print(f"Required minimum: 40%")