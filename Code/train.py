"""
MAI/IDL SS26 - Final assignment.

MG 6/6/2026
"""
import json

import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loaders
import models
from fit import Trainer


def main():
    # BUG FIX: config.json was completely missing — script crashed immediately on startup
    # created config.json from scratch with SHARED settings and CONFIGS list
    with open("config.json", "r") as f:
        config = json.load(f)

    # SHARED contains settings common to all runs — batch size, learning rate etc
    shared = config["SHARED"]
    # CONFIGS is a list of all model/dataset combinations to run
    configs = config["CONFIGS"]

    # BUG FIX: original only checked for CUDA and fell back to CPU
    # added MPS detection so Apple Silicon GPU is used — 3-5x faster on Mac
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training executing on device: {device}\n")

    for cfg in configs:
        print(f"{'='*60}")
        print(f"  Model: {cfg['MODEL']}  |  Dataset: {cfg['DATA']}")
        print(f"{'='*60}")

        # BUG FIX: original used _ to discard test_loader — final evaluation was impossible
        # now saved as test_loader and used after training to get final test accuracy
        train_loader, val_loader, test_loader = get_loaders(data=cfg["DATA"],data_path=shared["DATA_PATH"],batch_size=shared["BATCH_SIZE"])

        model_class = getattr(models, cfg["MODEL"])
        # BUG FIX: drop_rate was hardcoded as 0.99 — disabled 99% of neurons
        # changed to read from config where correct value 0.5 is set
        # also removed invalid activation_str=None kwarg that was being passed
        model = model_class(in_channels=cfg["CHANNELS"],num_classes=cfg["NUM_CLASSES"],drop_rate=shared["DROP_RATE"]).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=shared["LEARNING_RATE"])

        trainer = Trainer(model, criterion, optimizer, device)
        trainer.fit(train_loader, val_loader, epochs=cfg["EPOCHS"])

        # BUG FIX: original had no test evaluation at all after training
        # added evaluate on test set to get final real-world accuracy
        test_loss, test_acc = trainer.evaluate(test_loader)
        print(f"\nFinal Test Accuracy: {test_acc:.2f}%\n")


if __name__ == "__main__":
    main()