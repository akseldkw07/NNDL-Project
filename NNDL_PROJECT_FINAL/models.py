# Methods for Shared backbone + two heads + KL divergence
import torch
import torch.nn as nn
from resnet_pretrain import build_resnet_backbone


class SharedBackboneTwoHeads(nn.Module):
    def __init__(self, num_super, num_sub, backbone: str = "resnet18"):
        super().__init__()
        self.backbone, feat_dim = build_resnet_backbone(backbone)
        self.super_head = nn.Linear(feat_dim, num_super)
        self.sub_head = nn.Linear(feat_dim, num_sub)

    def forward(self, x):
        feats = self.backbone(x)
        super_logits = self.super_head(feats)
        sub_logits = self.sub_head(feats)
        return super_logits, sub_logits


# KL helper that maps subclass probs to superclass probs using sub_to_super mapping


def sub_probs_to_super_probs(sub_probs, sub_to_super, num_super):
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
    def __init__(self, num_classes, backbone: str = "resnet18"):
        super().__init__()
        self.backbone, feat_dim = build_resnet_backbone(backbone)
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        return logits


def accuracy_from_logits(logits, targets):
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct, total
