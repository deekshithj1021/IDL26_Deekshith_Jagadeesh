"""
MAI/IDL SS26 - Final assignment.

MG 6/6/2026
"""
import copy
import torch


class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        # BUG FIX: was "correct, sum = 0, 0" — sum shadows Python built-in sum()
        correct, total = 0, 0

        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # BUG FIX: zero_grad was missing — gradients were accumulating across batches
            # without this gradients explode and model cannot learn
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return running_loss / total, (correct / total) * 100

    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return running_loss / total, (correct / total) * 100

    def fit(self, train_loader, val_loader, epochs):
        print("\n Starting Training Routine...")
        print("-" * 50)

        # IMPROVEMENT: track best validation accuracy during training
        # first run showed final epoch weights were sometimes much worse than best epoch
        best_val_acc = 0.0
        best_weights = None

        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
                  f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")

            # save weights whenever validation accuracy improves
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # deepcopy makes independent copy — model can keep training without overwriting
                best_weights = copy.deepcopy(self.model.state_dict())

        # restore best weights so test evaluation uses best model not final epoch
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
            print(f" Restored best weights (val acc: {best_val_acc:.2f}%)")

        print("-" * 50)
        print("Training Complete!")