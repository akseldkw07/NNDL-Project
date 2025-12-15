import typing as t

import torch
from tqdm.contrib.logging import logging_redirect_tqdm

if t.TYPE_CHECKING:
    from .abc_nn import ABCNN


# Callback System
class TrainingCallback:
    """Base class for training callbacks. Override methods to inject custom behavior."""

    def on_train_begin(self, model: "ABCNN", logs: dict[str, t.Any] | None = None) -> None:
        """Called at the beginning of training."""

    def on_train_end(self, model: "ABCNN", logs: dict[str, t.Any] | None = None) -> None:
        """Called at the end of training."""

    def on_epoch_begin(self, epoch: int, model: "ABCNN", logs: dict[str, t.Any] | None = None) -> None:
        """Called at the beginning of each epoch."""

    def on_epoch_end(self, epoch: int, model: "ABCNN", logs: dict[str, t.Any] | None = None) -> None:
        """Called at the end of each epoch."""

    def on_batch_begin(self, batch: int, model: "ABCNN", logs: dict[str, t.Any] | None = None) -> None:
        """Called at the beginning of each batch."""

    def on_batch_end(self, batch: int, model: "ABCNN", logs: dict[str, t.Any] | None = None) -> None:
        """Called at the end of each batch."""


# Built-in Callbacks
class GradientClippingCallback(TrainingCallback):
    """Clips gradients by norm after each batch."""

    def __init__(self, max_norm: float):
        self.max_norm = max_norm

    def on_batch_end(self, batch: int, model: "ABCNN", logs: dict[str, t.Any] | None = None) -> None:
        if logs and logs.get("after_backward", False):
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)


class LearningRateSchedulerCallback(TrainingCallback):
    """Steps the learning rate scheduler."""

    def __init__(self, step_on: t.Literal["epoch", "batch"] = "epoch"):
        self.step_on = step_on

    def on_epoch_end(self, epoch: int, model: "ABCNN", logs: dict[str, t.Any] | None = None) -> None:
        if self.step_on == "epoch":
            model.scheduler.step()
            with logging_redirect_tqdm([model.logger]):
                model.logger.info(
                    f"Learning Rate: {model.optimizer.param_groups[0]['lr']:.6f}, Step: {epoch}, Epoch: {model.scheduler.last_epoch}"
                )

    def on_batch_end(self, batch: int, model: "ABCNN", logs: dict[str, t.Any] | None = None) -> None:
        if self.step_on == "batch":
            model.scheduler.step()
