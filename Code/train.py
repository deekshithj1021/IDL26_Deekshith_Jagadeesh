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
    with open("config.json", "r") as f:
        config = json.load(f)

    shared = config["SHARED"]
    configs = config["CONFIGS"]

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

        train_loader, val_loader, test_loader = get_loaders(data=cfg["DATA"],data_path=shared["DATA_PATH"],batch_size=shared["BATCH_SIZE"])

        model_class = getattr(models, cfg["MODEL"])
        model = model_class(in_channels=cfg["CHANNELS"],num_classes=cfg["NUM_CLASSES"],drop_rate=shared["DROP_RATE"]).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(),lr=shared["LEARNING_RATE"])

        trainer = Trainer(model, criterion, optimizer, device)
        trainer.fit(train_loader, val_loader, epochs=cfg["EPOCHS"])

        test_loss, test_acc = trainer.evaluate(test_loader)
        print(f"\nFinal Test Accuracy: {test_acc:.2f}%\n")


if __name__ == "__main__":
    main()