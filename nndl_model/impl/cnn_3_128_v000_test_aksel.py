import torch.nn as nn
from torch import Tensor

from nndl_model.base import BaseModel


class CNN_3_128_V000_Test_Aksel(BaseModel):
    def __init__(self, M: Tensor, lr: float = 1e-3, feature_dim: int = 64):
        super().__init__(M=M, lr=lr)  # 1) init base (no optimizer yet)
        self.model = nn.Sequential(  # 2) define backbone params
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),  # -> [B, 64]
        )
        # 3) add heads + optimizer now exists
        self.configure_hierarchy(feature_dim=feature_dim)
        self.reset_optimizer()
