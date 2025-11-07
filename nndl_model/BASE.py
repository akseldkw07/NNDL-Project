import torch
import torch.nn as nn
import wandb
from nndl_model.constants import DEVICE_TORCH_STR
import typing as t


class BaseModel(nn.Module):
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    device: t.Literal["cuda", "mps", "cpu"]

    def __init__(self):
        super().__init__()
        self.best_loss = float("inf")
        self.best_acc = 0.0
        self.epochs_trained = 0
        self.device = DEVICE_TORCH_STR
        self.to(self.device)

    def forward(self, x):
        """
        Forward pass placeholder to be implemented by subclasses.
        """
        raise NotImplementedError("Forward method not implemented.")

    def train_model(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: t.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        epochs=10,
    ):
        """
        Train the model using the provided data loaders, criterion, and optimizer.
        Logs training and validation metrics to wandb.
        TODO make use of double head prediction and appropriate loss

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            criterion (nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer.
            device (torch.device): Device to run training on.
            epochs (int): Number of epochs to train.

        """
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            running_corrects = 0
            total = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data).item()
                total += inputs.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total

            val_loss, val_acc = self.evaluate(val_loader, criterion, self.device)

            wandb.log(
                {
                    "Epoch": self.epochs_trained + epoch + 1,
                    "Train Loss": epoch_loss,
                    "Train Accuracy": epoch_acc,
                    "Validation Loss": val_loss,
                    "Validation Accuracy": val_acc,
                }
            )

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_acc = val_acc
                self.save_weights("best_model.pth")

            print(
                f"Epoch {self.epochs_trained + epoch + 1}/{self.epochs_trained + epochs} "
                f"Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.4f} "
                f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}"
            )

        self.epochs_trained += epochs

    def evaluate(self, val_loader, criterion, device):
        """
        Evaluate the model on a validation set.

        Args:
            val_loader (DataLoader): DataLoader for validation data.
            criterion (nn.Module): Loss function.
            device (torch.device): Device to run evaluation on.

        Returns:
            tuple: (validation loss, validation accuracy)
        """
        self.eval()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = self(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data).item()
                total += inputs.size(0)

        val_loss = running_loss / total
        val_acc = running_corrects / total

        return val_loss, val_acc

    def summary(self):
        """
        Returns a string summarizing training progress and best metrics.
        """
        return (
            f"Epochs trained: {self.epochs_trained}\n"
            f"Best Validation Loss: {self.best_loss:.4f}\n"
            f"Best Validation Accuracy: {self.best_acc:.4f}"
        )

    def save_weights(self, path):
        """
        Save the model weights to a file.

        Args:
            path (str): File path to save the weights.
        """
        torch.save(self.state_dict(), path)

    def load_weights(self, path, map_location=None):
        """
        Load model weights from a file.

        Args:
            path (str): File path from which to load weights.
            map_location (torch.device or str, optional): Device mapping for loading.
        """
        self.load_state_dict(torch.load(path, map_location=map_location))
