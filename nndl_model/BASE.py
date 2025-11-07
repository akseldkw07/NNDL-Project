import pathlib
import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

from nndl_model.constants import DEVICE_TORCH_STR, MODEL_WEIGHT_DIR


class TorchDict(t.TypedDict):
    super: torch.Tensor
    sub: torch.Tensor


class BaseModel(nn.Module):
    lr: float
    best_loss: float
    best_acc: float
    epochs_trained: int
    model: nn.Module  # nn.Sequential or other nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    device: t.Literal["cuda", "mps", "xpu", "cpu"]  # DEVICE_TORCH_STR
    path: pathlib.Path
    criterion: nn.Module

    head_super: t.Optional[nn.Module]
    head_sub: t.Optional[nn.Module]
    M: t.Optional[torch.Tensor]  # [S, K] mapping buffer
    alpha: float
    beta: float
    gamma: float

    def __init__(self, lr: float = 0.001):
        super().__init__()
        self.lr = lr
        self.best_loss = float("inf")
        self.best_acc = 0.0
        self.epochs_trained = 0
        self.model = nn.Sequential()
        self.head_super = None
        self.head_sub = None
        # self.M = None
        self.alpha, self.beta, self.gamma = 1.0, 1.0, 0.1
        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer # NOTE: NOT SET
        # self.scheduler # NOTE: NOT SET
        self.device = DEVICE_TORCH_STR
        self.to(self.device)
        self.path = MODEL_WEIGHT_DIR / f"{self.__class__.__name__}_weights.pth"

    def reset_optimizer(self) -> None:
        """(Re)create optimizer and scheduler now that parameters exist."""
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

    def configure_hierarchy(
        self,
        feature_dim: int,
        num_super: int,
        num_sub: int,
        M: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.1,
    ):
        """
        Turn on hierarchical classification.
        - feature_dim: size of features emitted by self.model(x)
        - num_super: number of super classes (S)
        - num_sub: number of sub classes (K)
        - M: binary/float mapping tensor of shape [S, K] with M[s,k]=1 iff sub k belongs to super s
        - alpha, beta, gamma: loss weights for super CE, sub CE, and consistency KL
        """
        self.head_super = nn.Linear(feature_dim, num_super)
        self.head_sub = nn.Linear(feature_dim, num_sub)

        # Register mapping as non-trainable buffer on the correct device
        M = M.to(self.device).float()
        if M.dim() != 2 or M.size(0) != num_super or M.size(1) != num_sub:
            raise ValueError("M must have shape [num_super, num_sub]")
        self.register_buffer("M", M)  # saves/loads with state_dict, not optimized

        self.alpha, self.beta, self.gamma = float(alpha), float(beta), float(gamma)

        # Recreate optimizer so new head params are included
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer.param_groups[0]["lr"])

    def forward(self, x):
        h = self.model(x)
        if self.head_super is not None and self.head_sub is not None:
            return self.head_super(h), self.head_sub(h)
        return h

    def _hierarchical_loss(
        self, logits_sup: torch.Tensor, logits_sub: torch.Tensor, y_sup: torch.Tensor, y_sub: torch.Tensor
    ):
        if self.M is None:
            raise RuntimeError("Hierarchy not configured. Call configure_hierarchy(...) first.")

        loss_sup = F.cross_entropy(logits_sup, y_sup)
        loss_sub = F.cross_entropy(logits_sub, y_sub)

        # Probabilities
        p_sup = F.softmax(logits_sup, dim=1)
        p_sub = F.softmax(logits_sub, dim=1)

        # Aggregate subclass probabilities to super classes: tilde_p_sup = normalize(M @ p_sub)
        tilde_p_sup = (self.M @ p_sub.T).T  # [B, S]
        tilde_p_sup = tilde_p_sup / tilde_p_sup.sum(dim=1, keepdim=True).clamp_min(1e-12)

        # KL(tilde || p_sup)
        kl = F.kl_div(p_sup.log(), tilde_p_sup, reduction="batchmean")

        return self.alpha * loss_sup + self.beta * loss_sub + self.gamma * kl

    def get_loss(self, outputs: tuple[torch.Tensor, torch.Tensor], labels: TorchDict | torch.Tensor):
        device = self.device
        if isinstance(outputs, tuple) and isinstance(labels, dict):
            logits_sup, logits_sub = outputs
            y_sup = labels["super"].to(device)
            y_sub = labels["sub"].to(device)
            return self._hierarchical_loss(logits_sup, logits_sub, y_sup, y_sub)
        elif isinstance(labels, torch.Tensor):
            logits = outputs if not isinstance(outputs, tuple) else outputs[0]
            labels = labels.to(device)
            return self.criterion(logits, labels)
        else:
            raise ValueError("Outputs and labels format do not match for loss computation.")

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 10) -> None:
        """
        Train the model using the provided data loaders and optimizer.
        Logs training and validation metrics to wandb.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            epochs (int): Number of epochs to train.

        """
        device = self.device
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            running_corrects_sup = 0
            running_corrects_sub = 0
            total = 0

            for inputs, labels in train_loader:
                inputs: torch.Tensor = inputs.to(device)
                labels: torch.Tensor | TorchDict
                if isinstance(labels, dict):
                    labels_sup = labels["super"].to(device)
                    labels_sub = labels["sub"].to(device)
                else:
                    labels_sup = labels_sub = labels.to(device)

                self.optimizer.zero_grad()
                outputs = self(inputs)

                loss = self.get_loss(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                # Accuracy calculation
                if isinstance(outputs, tuple) and isinstance(labels, dict):
                    logits_sup, logits_sub = outputs
                    _, preds_sup = torch.max(logits_sup, 1)
                    _, preds_sub = torch.max(logits_sub, 1)
                    running_corrects_sup += (preds_sup == labels_sup).sum().item()
                    running_corrects_sub += (preds_sub == labels_sub).sum().item()
                else:
                    logits = outputs if not isinstance(outputs, tuple) else outputs[0]
                    _, preds = torch.max(logits, 1)
                    running_corrects_sup += (preds == labels_sup).sum().item()
                    running_corrects_sub += (preds == labels_sub).sum().item()

                total += inputs.size(0)

            epoch_loss = running_loss / total
            epoch_acc_sup = running_corrects_sup / total
            epoch_acc_sub = running_corrects_sub / total

            val_loss, val_acc_sub, val_acc_sup = self.evaluate(val_loader)

            wandb.log(
                {
                    "Epoch": self.epochs_trained + epoch + 1,
                    "Train Loss": epoch_loss,
                    "Validation Loss": val_loss,
                    "Train Accuracy (super)": epoch_acc_sup,
                    "Train Accuracy (sub)": epoch_acc_sub,
                    "Validation Accuracy (super)": val_acc_sup,
                    "Validation Accuracy (sub)": val_acc_sub,
                }
            )

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_acc = val_acc_sub
                self.save_weights()

            print(
                f"Epoch {self.epochs_trained + epoch + 1}/{self.epochs_trained + epochs} "
                f"Train Loss: {epoch_loss:.4f} "
                f"Train Acc (sup/sub): {epoch_acc_sup:.4f}/{epoch_acc_sub:.4f} "
                f"Val Loss: {val_loss:.4f} "
                f"Val Acc (sup/sub): {val_acc_sup:.4f}/{val_acc_sub:.4f}"
            )

        self.epochs_trained += epochs

    def evaluate(self, val_loader: DataLoader) -> tuple[float, float, float]:
        """
        Evaluate the model on a validation set.

        Args:
            val_loader (DataLoader): DataLoader for validation data.
        Returns:
            tuple: (validation loss, validation accuracy (subclass), validation accuracy (superclass))
        """
        self.eval()
        running_loss = 0.0
        running_corrects_sup = 0
        running_corrects_sub = 0
        total = 0
        device = self.device

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs: torch.Tensor = inputs.to(device)
                labels: torch.Tensor | TorchDict = labels
                if isinstance(labels, dict):
                    labels_sup = labels["super"].to(device)
                    labels_sub = labels["sub"].to(device)
                else:
                    labels_sup = labels_sub = labels.to(device)

                outputs = self(inputs)

                loss = self.get_loss(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

                if isinstance(outputs, tuple) and isinstance(labels, dict):
                    logits_sup, logits_sub = outputs
                    _, preds_sup = torch.max(logits_sup, 1)
                    _, preds_sub = torch.max(logits_sub, 1)
                    running_corrects_sup += (preds_sup == labels_sup).sum().item()
                    running_corrects_sub += (preds_sub == labels_sub).sum().item()
                else:
                    logits = outputs if not isinstance(outputs, tuple) else outputs[0]
                    _, preds = torch.max(logits, 1)
                    running_corrects_sup += (preds == labels_sup).sum().item()
                    running_corrects_sub += (preds == labels_sub).sum().item()

                total += inputs.size(0)

        val_loss = running_loss / total
        val_acc_sup = running_corrects_sup / total
        val_acc_sub = running_corrects_sub / total

        return val_loss, val_acc_sub, val_acc_sup

    def summary(self):
        """
        Returns a string summarizing training progress and best metrics.
        """
        return (
            f"Epochs trained: {self.epochs_trained}\n"
            f"Best Validation Loss: {self.best_loss:.4f}\n"
            f"Best Validation Accuracy: {self.best_acc:.4f}"
        )

    def save_weights(self):
        """
        Save the model weights to a file.

        Args:
            path (str): File path to save the weights.
        """
        torch.save(self.state_dict(), self.path)

    def load_weights(self):
        """
        Load model weights from a file.

        Args:
            path (str): File path from which to load weights.
            map_location (torch.device or str, optional): Device mapping for loading.
        """
        self.load_state_dict(torch.load(self.path, map_location=self.device))
