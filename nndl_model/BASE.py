import pathlib
import torch
import torch.nn as nn
import wandb
from nndl_model.constants import DEVICE_TORCH_STR, MODEL_WEIGHT_DIR
import typing as t
import torch.nn.functional as F


class BaseModel(nn.Module):
    best_loss: float
    best_acc: float
    epochs_trained: int
    model: nn.Module  # nn.Sequential or other nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    device: t.Literal["cuda", "mps", "xpu", "cpu"]  # DEVICE_TORCH_STR
    path: pathlib.Path
    criteria: t.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    head_super: t.Optional[nn.Module]
    head_sub: t.Optional[nn.Module]
    M: t.Optional[torch.Tensor]  # [S, K] mapping buffer
    alpha: float
    beta: float
    gamma: float

    def __init__(self, lr: float = 0.001):
        super().__init__()
        self.best_loss = float("inf")
        self.best_acc = 0.0
        self.epochs_trained = 0
        self.model = nn.Sequential()
        self.head_super = None
        self.head_sub = None
        self.M = None
        self.alpha, self.beta, self.gamma = 1.0, 1.0, 0.1
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        self.device = DEVICE_TORCH_STR
        self.to(self.device)
        self.path = MODEL_WEIGHT_DIR / f"{self.__class__.__name__}_weights.pth"

    def configure_hierarchy(
        self,
        feature_dim: int,
        num_super: int,
        num_sub: int,
        M: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.1,
    ) -> None:
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
        self,
        logits_sup: torch.Tensor,
        logits_sub: torch.Tensor,
        y_sup: torch.Tensor,
        y_sub: torch.Tensor,
    ) -> torch.Tensor:
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

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            criterion (nn.Module): Loss function.
            epochs (int): Number of epochs to train.

        """
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            running_corrects = 0
            total = 0

            for inputs, labels in train_loader:
                inputs: torch.Tensor
                labels: torch.Tensor
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self(inputs)

                # Unpack labels: support (y_sup, y_sub) tuple/list or dict with keys 'super'/'sub'
                y_sup = y_sub = None
                if isinstance(labels, (tuple, list)) and len(labels) == 2:
                    y_sup, y_sub = labels
                elif isinstance(labels, dict) and "super" in labels and "sub" in labels:
                    y_sup, y_sub = labels["super"], labels["sub"]

                if isinstance(outputs, tuple) and y_sup is not None and y_sub is not None:
                    logits_sup, logits_sub = outputs
                    loss = self._hierarchical_loss(logits_sup, logits_sub, y_sup.to(self.device), y_sub.to(self.device))
                    # Accuracy: super and sub separately
                    _, preds_sup = torch.max(logits_sup, 1)
                    _, preds_sub = torch.max(logits_sub, 1)
                    running_corrects += (preds_sub == y_sub).sum().item()
                else:
                    # Fallback: single-head criterion
                    logits = outputs if not isinstance(outputs, tuple) else outputs[0]
                    loss = criterion(logits, labels)
                    _, preds = torch.max(logits, 1)
                    running_corrects += torch.sum(preds == labels.data).item()

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
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
                self.save_weights()

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
                inputs: torch.Tensor
                labels: torch.Tensor
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self(inputs)

                y_sup = y_sub = None
                labels_in = labels
                if isinstance(labels_in, (tuple, list)) and len(labels_in) == 2:
                    y_sup, y_sub = labels_in
                elif isinstance(labels_in, dict) and "super" in labels_in and "sub" in labels_in:
                    y_sup, y_sub = labels_in["super"], labels_in["sub"]

                if isinstance(outputs, tuple) and y_sup is not None and y_sub is not None:
                    logits_sup, logits_sub = outputs
                    loss = self._hierarchical_loss(logits_sup, logits_sub, y_sup.to(device), y_sub.to(device))
                    _, preds_sub = torch.max(logits_sub, 1)
                    running_corrects += (preds_sub == y_sub.to(device)).sum().item()
                else:
                    logits = outputs if not isinstance(outputs, tuple) else outputs[0]
                    loss = criterion(logits, labels)
                    _, preds = torch.max(logits, 1)
                    running_corrects += torch.sum(preds == labels.data).item()

                running_loss += loss.item() * inputs.size(0)
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
