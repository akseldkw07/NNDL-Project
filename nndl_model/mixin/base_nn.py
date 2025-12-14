from __future__ import annotations

import json
import logging
import os
import pathlib
import pprint
import typing as t
from copy import deepcopy
from dataclasses import asdict

import torch
import torch.nn as nn
import wandb
from mixin.config import TrainingConfig
from model_utils.weights import re_init_weights
from torch.utils.data import DataLoader
from tqdm.contrib.logging import logging_redirect_tqdm

from ..constants import DEVICE_TORCH_STR, MODEL_WEIGHT_DIR
from .abc_nn import ABCNN, Eval_HyperParams_Data, HyperParamConfig, ModelEvalState, ModelPathVals
from .batch_loader import XTYPE, YTYPE, make_loader_from_xy
from .config import PriorConfig

LOAD_LTRL = t.Literal["assert", "try", "fresh"]

# Default training state
DEFAULT_MODEL_STATE = ModelEvalState(
    best_eval_loss=float("inf"),
    best_eval_r2=float("-inf"),
    best_eval_accuracy=float("-inf"),
    best_eval_f1=float("-inf"),
    epochs_trained=0,
)


class BaseNN(ABCNN, nn.Module):
    # region INHERITED METHODS
    """INHERITED METHODS"""

    def __init__(
        self,
        hparams: HyperParamConfig | None = None,
        prior_config: PriorConfig | None = None,
        training_config: TrainingConfig | None = None,
    ):
        super().__init__()

        # Initialize logging and model components
        self.logger = logging.getLogger(self.name)

        if not self.logger.handlers:
            handler = logging.StreamHandler()  # or StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                fmt="[%(asctime)s | %(levelname)-8s | %(name)s ] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.device = DEVICE_TORCH_STR

        # Initialize default Hyperparameters (may be overridden by load)
        self.hparams = hparams or HyperParamConfig()
        self.training_config = training_config or TrainingConfig()
        self.prior_config = prior_config or PriorConfig()

        # Initialize default training state (may be overridden by load)
        self.eval_best = DEFAULT_MODEL_STATE.copy()

        self.to(self.device)

    def post_init(self, load_weights: LOAD_LTRL = "try", override_hp_from_file: bool = False) -> None:
        """
        Specify actions that must be taken after the model architecture is defined.

        1) Resetting the optimizer
        2) Loading saved weights
        3) Restoring model performance
        4) Set prior
        5) Logging full state
        """
        self._reset_optimizer()

        self._load_weights_act = load_weights
        self.load_weights(load_weights, override_hp_from_file)

        self._post_init_done = True

        self.to(self.device)

        # Only log initial state in verbose mode
        if self.training_config.log_on_improvement:
            self.logger.info("📊 Initial State:")
            pprint.pprint(self.Model_Eval_HyperParams_Data_Display)

    def _reset_optimizer(self) -> None:
        """(Re)create optimizer and scheduler now that parameters exist."""
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.hparams.stepsize, gamma=self.hparams.gamma
        )

    # SAVING
    @property
    def root_dir(self) -> pathlib.Path:
        return MODEL_WEIGHT_DIR / f"{self.name}"

    @property
    def model_paths(self):
        ret = ModelPathVals(
            model_path=self.root_dir / "model.txt",
            eval_hparams_path=self.root_dir / "eval-hparams.json",
            weight_path=self.root_dir / "weights.pt",
        )
        return ret

    @property
    def Model_Eval_HyperParams_Data(self):
        return Eval_HyperParams_Data(state=self.eval_best, hparams=self.hparams)

    @property
    def Model_Eval_HyperParams_Data_Display(self):
        ret = deepcopy(self.Model_Eval_HyperParams_Data)
        for k, v in ret.state.items():
            if not isinstance(v, float):
                continue
            if v < 1e-3 and v > 0:
                ret.state[k] = f"{v:.2e}"
            if v <= 1:
                ret.state[k] = f"{v:.2%}"
            if v >= 1:
                ret.state[k] = round(v, 3)
        return ret

    def save_weights(self, increment_version: bool = False):
        """
        Save the model weights, summary, and training state to a file.
        """
        if increment_version:
            version_num = int(self.nickname.split("v")[-1]) + 1
            self.nickname = f"v{version_num:03d}"
            self.logger.warning(f"Incremented model nickname to {self.nickname}.")

        if self.training_config.log_on_improvement:
            with logging_redirect_tqdm([self.logger]):
                self.logger.info(f"💾 Saved weights to {self.root_dir.name}")

        self.root_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), self.model_paths.weight_path)
        with open(self.model_paths.model_path, "w", encoding="utf-8") as f:
            f.write(self._summary())

        with open(self.model_paths.eval_hparams_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.Model_Eval_HyperParams_Data), f)

    def load_weights(self, load_weights: LOAD_LTRL = "try", override_hp_from_file: bool = False):
        """
        Load the model weights, summary, and training state from a file.

        Args:
            load_weights: Controls loading behavior - "assert" (must succeed), "try" (silent fail), or "fresh" (skip loading).
            override_load_hp: If True, replace current hyperparameters with loaded values from the state file.
        """
        if load_weights == "assert":
            self._load_weights(override_hp_from_file)
        elif load_weights == "try":
            try:
                self._load_weights(override_hp_from_file)
            except Exception as ex:
                self.logger.error(f"Failed to load state from {self.root_dir}: {ex}. Continuing with fresh weights.")
                self.eval_best = DEFAULT_MODEL_STATE.copy()
        elif load_weights == "fresh":
            self.eval_best = DEFAULT_MODEL_STATE.copy()
        else:
            raise ValueError("Invalid load_weights value.")

    def _load_weights(self, override_load_hp):
        """
        Attempt to load model weights and state from specified path.
        """
        assert os.path.exists(self.model_paths.weight_path), (
            f"Model weights file not found at {self.model_paths.weight_path}. "
            "Ensure the model has been trained and saved before loading."
        )
        with open(self.model_paths.eval_hparams_path, encoding="utf-8") as f:
            full_state = json.load(f)

        self.load_state_dict(torch.load(self.model_paths.weight_path, map_location=self.device))

        self.eval_best = self.eval_best | full_state["state"]
        if override_load_hp:
            self.hparams = self.hparams | full_state["hparams"]

        self.logger.info(f"Loaded model weights and state from {self.root_dir}.")

    def reset_weights(self) -> None:
        """
        Reinitialize model weights to their default state.

        This applies appropriate initialization based on layer type:
        - Linear/Conv layers: Kaiming uniform initialization
        - Embedding layers: Normal initialization
        - BatchNorm/LayerNorm: Reset to default (weight=1, bias=0)
        """

        # Apply initialization to all modules
        self.apply(re_init_weights)

        # Reset optimizer to forget momentum/state
        self._reset_optimizer()

        # Reset training state
        self.eval_best = DEFAULT_MODEL_STATE.copy()

        self.logger.info("✓ Reset all model weights to initial state")

    # NAMING
    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}_{self.nickname}"

    @classmethod
    def clsname(cls, include_version: bool = True) -> str:
        """
        Returns a string identifier combining the class name and version.
        """
        version = getattr(cls, "version", "v000")
        if include_version:
            return f"{cls.__name__}_{version}"
        return cls.__name__

    def _summary(self):
        """
        Prints a string summarizing training progress and best metrics.
        """
        name = self.name
        model = self.__str__()
        linesep = "-" * 100
        return (
            f"Model: {name=}\n{linesep}\n{model}\n\n"
            f"{linesep}\n{pprint.pformat(self.Model_Eval_HyperParams_Data_Display)}\n\n"
            f"{linesep}\n{pprint.pformat(self.prior_config)}\n"
            f"{linesep}\n{pprint.pformat(self.training_config)}\n"
        )

    def summary(self):
        print(self._summary())

    # endregion
    # region LIKELY REDEFINED IN SUBCLASS
    """LIKELY REDEFINED IN SUBCLASS"""

    @property
    def criterion(self):
        return self._criterion.to(self.device)

    def get_loss(self, outputs: torch.Tensor, labels: torch.Tensor, inputs: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute total loss. If in training mode, includes prior regularization.
        """
        # Base loss (cross-entropy, MSE, etc.)
        base_loss = self.criterion(outputs, labels)

        # Prior regularization
        prior_loss = torch.tensor(0.0, device=self.device)
        if self.training:
            # Compute prior loss only during training
            if self.training_config.log_if_calc_prior_loss:
                with logging_redirect_tqdm([self.logger]):
                    self.logger.info("Computing prior loss...")
            prior_loss = self.compute_prior_loss(inputs=inputs, outputs=outputs)

        return base_loss + prior_loss

    def _patience_reached(self, epochs_no_improve: int) -> bool:
        if self.training_config.early_stopping_patience is None:
            return False
        patience_reached = epochs_no_improve >= self.training_config.early_stopping_patience
        if patience_reached:
            with logging_redirect_tqdm([self.logger]):
                self.logger.warning(
                    f"Early stopping activated: no improvement for {self.training_config.early_stopping_patience} consecutive epochs."
                )
        return patience_reached

    def _to_dataloader(self, data_: tuple[XTYPE, YTYPE] | DataLoader, batch_size: int) -> DataLoader:
        """
        Create DataLoader from data or ensure correct batch size.
        """
        data = data_ if isinstance(data_, DataLoader) else make_loader_from_xy(*data_, batch_size=batch_size)
        if data.batch_size != batch_size:
            with logging_redirect_tqdm([self.logger]):
                self.logger.warning(
                    f"Provided DataLoader has batch size {data.batch_size}, but expected {batch_size}. Using provided batch size {batch_size=}."
                )
            data = DataLoader(
                data.dataset,
                batch_size=batch_size,
                shuffle=getattr(data, "shuffle", True),
                num_workers=data.num_workers,
                drop_last=data.drop_last,
            )
        return data

    def _log_wandb(self, data: dict[str, float]):
        def format_key(key: str) -> str:
            return key.replace("_", " ").capitalize()

        if self.training_config.wandb:
            data = {format_key(k): v for k, v in data.items()}
            wandb.log(data)

    # endregion
