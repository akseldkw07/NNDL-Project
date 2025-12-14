import torch
import torch.nn as nn
import torch.nn.functional as F
from attr import dataclass

from nndl_model.mixin.abc_nn import HyperParamConfig
from nndl_model.mixin.config import PriorConfig, TrainingConfig
from nndl_model.nn_concrete import ClassificationNN


@dataclass
class CNN_Image_Config:
    M: torch.Tensor


class CNN_3_128_Test_Aksel(ClassificationNN):
    head_super: nn.Module  # NOTE: NOT SET in __init__
    head_sub: nn.Module  # NOTE: NOT SET in __init__
    _M: torch.Tensor  # [S, K] mapping buffer
    M: torch.Tensor  # this will be registered as buffer. This maps sub->super classes
    S: int
    K: int

    def __init__(
        self,
        model_cfg: CNN_Image_Config,
        hparams: HyperParamConfig | None = None,
        prior_config: PriorConfig | None = None,
        training_config: TrainingConfig | None = None,
    ):
        super().__init__(hparams, prior_config=prior_config, training_config=training_config)
        self.cfg = model_cfg

        # Register mapping as non-trainable buffer on the correct device
        self._M = self.cfg.M.to(self.device).float()
        self.S, self.K = self.cfg.M.shape

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),  # -> [B, 64]
        )

    def _configure_hierarchy(self, feature_dim: int):
        """
        Turn on hierarchical classification.
        - feature_dim: size of features emitted by self.model(x)
        - num_super: number of super classes (S)
        - num_sub: number of sub classes (K)
        - M: binary/float mapping tensor of shape [S, K] with M[s,k]=1 iff sub k belongs to super s
        - alpha, beta, gamma: loss weights for super CE, sub CE, and consistency KL
        """
        self.head_super = nn.Linear(feature_dim, self.S).to(self.device)
        self.head_sub = nn.Linear(feature_dim, self.K).to(self.device)

        if self._M.dim() != 2 or self._M.size(0) != self.S or self._M.size(1) != self.K:
            raise ValueError("M must have shape [S, K]")
        self.register_buffer("M", self._M)  # saves/loads with state_dict, not optimized

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h: torch.Tensor = self.model(x)
        return self.head_super(h), self.head_sub(h)

    def _hierarchical_loss(
        self, logits_sup: torch.Tensor, logits_sub: torch.Tensor, y_sup: torch.Tensor, y_sub: torch.Tensor
    ) -> torch.Tensor:
        # Cross-entropy wants raw logits (no softmax)
        loss_sup = F.cross_entropy(logits_sup, y_sup)
        loss_sub = F.cross_entropy(logits_sub, y_sub)

        # KL(tilde || p_sup), computed in log-prob space for stability
        log_p_sup = F.log_softmax(logits_sup, dim=1)  # [B, S]

        p_sub = F.softmax(logits_sub, dim=1)  # [B, K]
        eps = 1e-8
        tilde_p_sup = (self.M @ p_sub.T).T  # [B, S]
        tilde_p_sup = tilde_p_sup / tilde_p_sup.sum(dim=1, keepdim=True).clamp_min(eps)
        log_tilde_p_sup = torch.log(tilde_p_sup.clamp_min(eps))

        kl = F.kl_div(log_p_sup, log_tilde_p_sup, reduction="batchmean", log_target=True)

        return self.hparams["alpha"] * loss_sup + self.hparams["beta"] * loss_sub + self.hparams["gamma"] * kl

    def get_loss(self, outputs: tuple[torch.Tensor, torch.Tensor], labels: TorchDict | torch.Tensor) -> torch.Tensor:
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

    def evaluate(self, val_loader: DataLoader) -> tuple[float, float, float]:
        """
        Evaluate the model on a validation set.

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
