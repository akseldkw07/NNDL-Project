# Methods for Shared backbone + two heads + KL divergence
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet_pretrain import build_resnet_backbone


class CosineClassifier(nn.Module):
    """
    Cosine-similarity classifier (normalized linear head).
    logits = scale * cos(theta), where cos(theta)=<normalize(x), normalize(W)>.
    """

    def __init__(self, in_features: int, out_features: int, scale: float = 30.0, learn_scale: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        if learn_scale:
            self.scale = nn.Parameter(torch.tensor(float(scale)))
        else:
            self.register_buffer("scale", torch.tensor(float(scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, p=2, dim=1)  # [B, D]
        w = F.normalize(self.weight, p=2, dim=1)  # [C, D]
        logits = x @ w.t()  # [B, C]
        return logits * self.scale


class SharedBackboneTwoHeads(nn.Module):
    def __init__(
        self,
        num_super: int,
        num_sub: int,
        backbone: str = "resnet50",
        sub_head_type: str = "cosine",  # "linear" or "cosine"
        cosine_scale: float = 30.0,
        learn_scale: bool = True,
    ):
        super().__init__()
        self.backbone, feat_dim = build_resnet_backbone(backbone)
        self.super_head = nn.Linear(feat_dim, num_super)

        if sub_head_type == "cosine":
            sub_head = nn.Sequential(
                CosineClassifier(
                    in_features=feat_dim,
                    out_features=(out := num_sub),
                    scale=cosine_scale,
                    learn_scale=learn_scale,
                ),
                nn.ReLU(),
                nn.Linear(out, num_sub),
            )
            self.sub_head = sub_head
        elif sub_head_type == "linear":
            self.sub_head = nn.Linear(feat_dim, num_sub)
        else:
            raise ValueError(f"Unknown sub_head_type: {sub_head_type}")

    def forward(self, x):
        feats = self.backbone(x)
        super_logits = self.super_head(feats)
        sub_logits = self.sub_head(feats)
        return super_logits, sub_logits


# KL helper that maps subclass probs to superclass probs using sub_to_super mapping


def sub_probs_to_super_probs(sub_probs: torch.Tensor, sub_to_super: dict, num_super: int) -> torch.Tensor:
    """
    sub_probs: (B, num_sub), softmax over subclasses
    returns: (B, num_super), summed probs per super-class
    """
    B, num_sub = sub_probs.shape
    super_probs = torch.zeros(B, num_super, device=sub_probs.device)

    for sub_idx, super_idx in sub_to_super.items():
        super_probs[:, super_idx] += sub_probs[:, sub_idx]

    # For safety: re-normalize in case of any numeric drift
    super_probs = super_probs / (super_probs.sum(dim=1, keepdim=True) + 1e-8)
    return super_probs


class SingleHeadModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet50",
        head: str = "linear",  # "linear" or "cosine"
        cosine_scale: float = 30.0,
        learn_scale: bool = True,
    ):
        super().__init__()
        self.backbone, feat_dim = build_resnet_backbone(backbone)

        if head == "cosine":
            self.head = nn.Sequential(
                CosineClassifier(
                    in_features=feat_dim,
                    out_features=(out := num_classes),
                    scale=cosine_scale,
                    learn_scale=learn_scale,
                ),
                nn.ReLU(),
                nn.Linear(out, num_classes),
            )
        elif head == "linear":
            self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor):
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct, total
