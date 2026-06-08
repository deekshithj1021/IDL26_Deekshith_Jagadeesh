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

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training executing on device: {device}")

    train_loader, val_loader, test_loader = get_loaders(data=config["DATA"], data_path=config["DATA_PATH"], batch_size=config["BATCH_SIZE"])  #added test_loader to get_loaders because we want to be able to evaluate on the test set after training

    model_class = getattr(models, config["MODEL"])
    model = model_class(in_channels=config["CHANNELS"], num_classes=config["NUM_CLASSES"], drop_rate=config["DROP_RATE"]).to(device)  # the droprate is changed from 0.99 to 0.5 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

    trainer = Trainer(model, criterion, optimizer, device)
    trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"])

    test_loss, test_acc = trainer.evaluate(test_loader) #test the model on the test set after trainingbut the test_loss is not used because we are only interested in the test accuracy for the final evaluation
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()