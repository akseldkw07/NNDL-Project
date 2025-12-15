# Backbone builder

# Using ImageNet pretrained ResNet backbone, and chopping off FC head to Transfer Learn

import os

import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models


def build_resnet_backbone(backbone: str):
    # Using torchvision ResNet-18 with ImageNet weights
    if backbone == "resnet18":
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif backbone == "resnet50":
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        raise ValueError(f"Unknown BACKBONE: {backbone}")
    # Remove the final classification layer
    in_features = base.fc.in_features
    base.fc = nn.Identity()  # type: ignore[assignment]
    return base, in_features


# Dataset functions
class BirdDogReptileDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["image"]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        super_idx = int(row["superclass_index"])
        sub_idx = int(row["subclass_index"])
        return image, super_idx, sub_idx


# Test dataset (for leaderboard predictions)


class BirdDogReptileTestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # assumes images are named 0.jpg, 1.jpg, ..., N-1.jpg
        self.filenames = sorted(os.listdir(img_dir), key=lambda x: int(os.path.splitext(x)[0]))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, img_name
