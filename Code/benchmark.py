"""
MAI/IDL SS26 - Final Assignment
Automated benchmark runner for all model/dataset combinations.
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

with open("config.json", "r") as f:
    config = json.load(f)

shared  = config["SHARED"]
configs = config["CONFIGS"]

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Running on device: {device}\n")

results = []

for cfg in configs:
    print(f"\n{'='*60}")
    print(f"  Model: {cfg['MODEL']}  |  Dataset: {cfg['DATA']}")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader = get_loaders(
        data=cfg["DATA"],
        data_path=shared["DATA_PATH"],
        batch_size=shared["BATCH_SIZE"]
    )

    model_class = getattr(models, cfg["MODEL"])
    model = model_class(
        in_channels=cfg["CHANNELS"],
        num_classes=cfg["NUM_CLASSES"],
        drop_rate=shared["DROP_RATE"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=shared["LEARNING_RATE"])
    trainer   = Trainer(model, criterion, optimizer, device)

    start = time.time()
    trainer.fit(train_loader, val_loader, epochs=cfg["EPOCHS"])
    train_time = round(time.time() - start, 1)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = model(images).max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    import numpy as np
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy  = round((all_preds == all_labels).mean() * 100, 2)
    precision = round(precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100, 2)
    recall    = round(recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100, 2)
    f1        = round(f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100, 2)
    params    = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dummy = torch.randn(1, cfg["CHANNELS"], 64, 64).to(device)
    model.eval()
    t = time.time()
    with torch.no_grad():
        for _ in range(100):
            model(dummy)
    latency = round((time.time() - t) / 100 * 1000, 2)

    print(f">>> Accuracy:   {accuracy:.2f}%")
    print(f">>> Precision:  {precision:.2f}%")
    print(f">>> Recall:     {recall:.2f}%")
    print(f">>> F1-Score:   {f1:.2f}%")
    print(f">>> Params:     {params:,}")
    print(f">>> Time:       {train_time:.1f}s")
    print(f">>> Latency:    {latency:.2f}ms/sample")

    results.append({
        "model": cfg["MODEL"], "dataset": cfg["DATA"],
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1,
        "params": params, "train_time": train_time, "latency": latency
    })

print(f"\n{'='*100}")
print(f"  FINAL BENCHMARK RESULTS")
print(f"{'='*100}")
print(f"{'Model':<12} {'Dataset':<10} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Params':>12} {'Time':>8} {'Latency':>10}")
print(f"{'-'*88}")
for r in results:
    print(f"{r['model']:<12} {r['dataset']:<10} "
          f"{r['accuracy']:>7.2f}% {r['precision']:>7.2f}% "
          f"{r['recall']:>7.2f}% {r['f1']:>7.2f}% "
          f"{r['params']:>12,} {r['train_time']:>7.1f}s {r['latency']:>9.2f}ms")
print(f"{'='*100}")