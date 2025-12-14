from __future__ import annotations

import logging
import pathlib
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .batch_loader import XTYPE, YTYPE
from .config import PriorConfig, TrainingConfig


@dataclass
class ModelPathVals:
    model_path: pathlib.Path
    eval_hparams_path: pathlib.Path
    weight_path: pathlib.Path


@dataclass
class HyperParamConfig:
    lr: float = 1e-3
    gamma: float = 0.5
    stepsize: int = 12


class ModelEvalState(t.TypedDict):
    epochs_trained: int
    best_eval_loss: float
    best_eval_r2: float
    best_eval_accuracy: float
    best_eval_f1: float


@dataclass
class Eval_HyperParams_Data:
    state: ModelEvalState
    hparams: HyperParamConfig


class ABCNN(ABC, nn.Module):
    nickname: str = "v000"  # eg. 'v001', 'L2reg-v000', ' Dropout-0.2v000'
    model: nn.Module  # nn.Sequential or other nn.Module TODO define in subclass
    optimizer: torch.optim.Optimizer  # NOTE: NOT SET in __init__
    scheduler: torch.optim.lr_scheduler.LRScheduler  # NOTE: NOT SET in __init__
    device: t.Literal["cuda", "mps", "xpu", "cpu"]
    _criterion: nn.Module  # TODO define

    hparams: HyperParamConfig
    training_config: TrainingConfig
    prior_config: PriorConfig

    _load_weights_act: t.Literal["assert", "try", "fresh"]
    _post_init_done: bool = False
    eval_best: ModelEvalState
    logger: logging.Logger

    @property
    @abstractmethod
    def root_dir(self) -> pathlib.Path: ...

    @property
    @abstractmethod
    def model_paths(self) -> ModelPathVals: ...

    @property
    @abstractmethod
    def Model_Eval_HyperParams_Data(self) -> Eval_HyperParams_Data: ...

    @property
    @abstractmethod
    def Model_Eval_HyperParams_Data_Display(self) -> Eval_HyperParams_Data: ...

    @abstractmethod
    def save_weights(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def load_weights(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def reset_weights(self, *args, **kwargs) -> None: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @classmethod
    @abstractmethod
    def clsname(cls, *args, **kwargs) -> str: ...

    @abstractmethod
    def summary(self, *args, **kwargs) -> None: ...

    @property
    @abstractmethod
    def criterion(self) -> nn.Module: ...

    @abstractmethod
    def post_init(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def set_prior(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def forward(self, *args, **kwargs) -> t.Any:
        """
        Note - don't call .to(device) here; bad for memory
        """
        ...

    @abstractmethod
    def compute_prior_loss(self, *args, **kwargs) -> torch.Tensor: ...

    @abstractmethod
    def get_loss(self, *args, **kwargs) -> torch.Tensor:
        """
        Compute total loss. If in training mode, includes prior regularization.
        """
        ...

    # Helper training methods
    @abstractmethod
    def _patience_reached(self, *args, **kwargs) -> bool: ...

    @abstractmethod
    def _to_dataloader(self, data_: tuple[XTYPE, YTYPE] | DataLoader, batch_size: int) -> DataLoader: ...

    @abstractmethod
    def _log_wandb(self, *args, **kwargs) -> None: ...

    @abstractmethod
    def predict(self, *args, **kwargs) -> dict[str, t.Any]: ...

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> dict[str, float]: ...

    @abstractmethod
    def train_model(self, *args, **kwargs) -> None: ...

    # Helper methods for training with early stopping

    @abstractmethod
    def _improved(self, *args, **kwargs) -> dict[str, bool]: ...

    @abstractmethod
    def _on_improvement(self, *args, **kwargs) -> int: ...
