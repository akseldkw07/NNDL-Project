# Backbone builder

# Using ImageNet pretrained ResNet backbone, and chopping off FC head to Transfer Learn

import torch.nn as nn
from torchvision import models


def build_resnet_backbone(backbone: str):
    # Using torchvision ResNet-18 with ImageNet weights
    if backbone == "resnet18":
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif backbone == "resnet50":
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unknown BACKBONE: {backbone}")
    # Remove the final classification layer
    in_features = base.fc.in_features
    base.fc = nn.Identity()
    return base, in_features
