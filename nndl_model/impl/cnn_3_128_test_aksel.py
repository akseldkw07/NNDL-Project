import typing as t

import torch.nn as nn
from torch import Tensor

from nndl_model.base_model import BaseModel, HyperParamDict


class CNN_3_128_Test_Aksel(BaseModel):
    def __init__(self, M: Tensor, feature_dim: int = 64, **hparams: t.Unpack[HyperParamDict]):
        super().__init__(M=M, **hparams)

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),  # -> [B, 64]
        )

        self.configure_hierarchy(feature_dim=feature_dim)
        self.reset_optimizer()
