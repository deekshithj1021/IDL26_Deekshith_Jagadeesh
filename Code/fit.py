"""
MAI/IDL SS26 - Final assignment.
MG 6/6/2026
"""
import copy
import torch


class Trainer:
    def __init__(self, model, criterion, optimizer, device, patience=5):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        self.best_val_acc = 0.0
        self.best_weights = None
        self.epochs_no_improve = 0

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

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

        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
                  f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")

            # Save best weights when val accuracy improves
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_weights = copy.deepcopy(self.model.state_dict())
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            # Stop early if no improvement for patience epochs
            if self.epochs_no_improve >= self.patience:
                print(f"\n Early stopping triggered at epoch {epoch+1} "
                      f"— no improvement for {self.patience} epochs")
                break

        # Restore the best weights found during training
        if self.best_weights is not None:
            self.model.load_state_dict(self.best_weights)
            print(f" Restored best weights (val acc: {self.best_val_acc:.2f}%)")

        print("-" * 50)
        print("Training Complete!")