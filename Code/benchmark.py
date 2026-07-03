"""
MAI/IDL SS26 - Final Assignment
Automated benchmark runner for all model/dataset combinations.
Tracks accuracy, precision, recall, F1-score and efficiency metrics
for the green initiative analysis and consolidated benchmark report.
"""
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import models
from data import get_loaders
from fit import Trainer


# ── Load config ─────────────────────────────────────────────────────────────
with open("config.json", "r") as f:
    config = json.load(f)

shared  = config["SHARED"]
configs = config["CONFIGS"]

# ── Device setup ────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Running on device: {device}\n")


# ── Helper: count trainable parameters ──────────────────────────────────────
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Helper: measure inference latency per sample ────────────────────────────
def measure_latency(model, channels, device, n=100):
    model.eval()
    dummy = torch.randn(1, channels, 64, 64).to(device)
    with torch.no_grad():
        start = time.time()
        for _ in range(n):
            model(dummy)
        elapsed = (time.time() - start) / n * 1000
    return round(elapsed, 2)


# ── Helper: compute precision, recall and F1 on test set ────────────────────
def compute_metrics(model, dataloader, device):
    model.eval()
    all_preds  = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall    = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1        = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    accuracy  = (all_preds == all_labels).mean() * 100

    return round(accuracy, 2), round(precision*100, 2), round(recall*100, 2), round(f1*100, 2)


# ── Results storage ─────────────────────────────────────────────────────────
results = []


# ── Main benchmark loop ─────────────────────────────────────────────────────
for cfg in configs:
    print(f"\n{'='*60}")
    print(f"  Model: {cfg['MODEL']}  |  Dataset: {cfg['DATA']}")
    print(f"{'='*60}")

    # Load data
    train_loader, val_loader, test_loader = get_loaders(
        data=cfg["DATA"],
        data_path=shared["DATA_PATH"],
        batch_size=shared["BATCH_SIZE"]
    )

    # Build model
    model_class = getattr(models, cfg["MODEL"])
    model = model_class(
        in_channels=cfg["CHANNELS"],
        num_classes=cfg["NUM_CLASSES"],
        drop_rate=shared["DROP_RATE"]
    ).to(device)

    num_params = count_params(model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=shared["LEARNING_RATE"])
    trainer   = Trainer(model, criterion, optimizer, device)

    # Train with timing
    train_start = time.time()
    trainer.fit(train_loader, val_loader, epochs=cfg["EPOCHS"])
    train_time  = round(time.time() - train_start, 1)

    # Compute all metrics on test set
    accuracy, precision, recall, f1 = compute_metrics(model, test_loader, device)

    # Measure inference latency
    latency = measure_latency(model, cfg["CHANNELS"], device)

    # Print results
    print(f">>> Test Accuracy:      {accuracy:.2f}%")
    print(f">>> Precision (macro):  {precision:.2f}%")
    print(f">>> Recall (macro):     {recall:.2f}%")
    print(f">>> F1-Score (macro):   {f1:.2f}%")
    print(f">>> Parameters:         {num_params:,}")
    print(f">>> Training Time:      {train_time:.1f}s")
    print(f">>> Inference Latency:  {latency:.2f}ms/sample")

    results.append({
        "model":      cfg["MODEL"],
        "dataset":    cfg["DATA"],
        "accuracy":   accuracy,
        "precision":  precision,
        "recall":     recall,
        "f1":         f1,
        "params":     num_params,
        "train_time": train_time,
        "latency":    latency
    })


# ── Final summary table ──────────────────────────────────────────────────────
print(f"\n{'='*100}")
print(f"  FINAL BENCHMARK RESULTS")
print(f"{'='*100}")
print(f"{'Model':<12} {'Dataset':<10} {'Accuracy':>10} {'Precision':>10} "
      f"{'Recall':>8} {'F1':>8} {'Params':>12} {'Time':>8} {'Latency':>10}")
print(f"{'-'*92}")
for r in results:
    print(f"{r['model']:<12} {r['dataset']:<10} "
          f"{r['accuracy']:>9.2f}% {r['precision']:>9.2f}% "
          f"{r['recall']:>7.2f}% {r['f1']:>7.2f}% "
          f"{r['params']:>12,} {r['train_time']:>7.1f}s "
          f"{r['latency']:>9.2f}ms")
print(f"{'='*100}")