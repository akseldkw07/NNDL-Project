from __future__ import annotations

import typing as t
from dataclasses import asdict, dataclass, field, fields

from .callback import GradientClippingCallback, LearningRateSchedulerCallback, TrainingCallback


# Training Configuration
@dataclass
class TrainingConfig:
    """Configuration for training loop behavior."""

    num_epochs: int = 10
    batch_size: int = 128

    # Gradient accumulation
    # Helps with larger batch sizes on limited memory by accumulating gradients over multiple steps
    gradient_accumulation_steps: int = 1
    max_grad_norm: float | None = None

    # Mixed precision training
    use_amp: bool = False

    # Evaluation and logging
    eval_every_n_epochs: int = 1
    eval_every_n_steps: int | None = None
    wandb_plot_every_n_batches: int | None = None
    early_stopping_patience: int | None = 25  # Stop if no improvement after this many epochs. None to disable.
    improvement_tol: float = 1e-4

    # Logging
    log_on_improvement: bool = True
    wandb: bool = True
    log_if_calc_prior_loss: bool = False  # For debugging - ensure we don't include prior loss in eval

    # Saving
    save_on_improvement: bool = True

    # Scheduler
    scheduler_step_on: t.Literal["epoch", "batch", "none"] = "epoch"

    # Progress bar
    show_progress: bool = True

    # Callbacks
    callbacks: list[TrainingCallback] = field(default_factory=list)

    def __post_init__(self):
        # Auto-add gradient clipping callback if specified
        if self.max_grad_norm is not None:
            self.callbacks.append(GradientClippingCallback(self.max_grad_norm))

        # Auto-add scheduler callback if specified
        if self.scheduler_step_on != "none":
            self.callbacks.append(LearningRateSchedulerCallback(self.scheduler_step_on))

    def __eq__(self, value: object) -> bool:
        """
        Check equality of two TrainingConfig instances.

        Special handling for 'callbacks' field to compare types and attributes.
        """
        diffs = {"self": {}, "other": {}}
        callbacks_diff = False
        if not isinstance(value, TrainingConfig):
            return False
        for f in fields(self):
            self_val = getattr(self, f.name)
            other_val = getattr(value, f.name)
            if f.name == "callbacks":
                if len(self_val) != len(other_val):
                    callbacks_diff = True
                for cb1, cb2 in zip(self_val, other_val):
                    if type(cb1) != type(cb2):
                        callbacks_diff = True
                    if cb1.__dict__ != cb2.__dict__:
                        callbacks_diff = True
            else:
                if self_val != other_val:
                    diffs["self"][f.name] = self_val
                    diffs["other"][f.name] = other_val

        self._diffs = diffs  # Store diffs for later retrieval

        if diffs["self"] or diffs["other"] or callbacks_diff:
            return False

        return True

    def get_diff(self, other: TrainingConfig):
        _ = self == other  # Populate self._diffs
        return self._diffs


def ret_updated_config(curr: TrainingConfig, new: TrainingConfig, **overrides) -> TrainingConfig:
    """
    Strip out None values from overrides and merge curr, new, and overrides into a new TrainingConfig.

    Return updated TrainingConfig with values from new and overrides.
    """
    pop_none = {"num_epochs", "batch_size"}

    overrides_safe = {}
    for k, v in overrides.items():
        if k in pop_none and v is None:
            continue
        elif v is None:
            raise ValueError(f"All non-hanlded override values must be non-None, got: {overrides=}.\n{pop_none=}")
        overrides_safe[k] = v

    data = asdict(curr)
    data.update(asdict(new))
    data.update(overrides_safe)

    # 🔑 Let __post_init__ rebuild callbacks from config; don't reuse old callback objects
    data.pop("callbacks", None)

    return TrainingConfig(**data)


@dataclass
class PriorConfig:
    """Configuration for prior regularization."""

    # ARD (Automatic Relevance Determination)
    use_ard: bool = False
    ard_weight: float = 1.0

    # Horseshoe prior
    use_horseshoe: bool = False
    horseshoe_tau: float = 0.1
    horseshoe_epsilon: float = 1e-8

    # Entropy regularization
    use_entropy: bool = False
    entropy_weight: float = 0.1

    # Smoothness prior (second derivative penalty)
    use_smoothness: bool = False
    smoothness_lambda: float = 0.01
    smoothness_samples: int = 1

    # L1 regularization (Lasso)
    use_l1: bool = False
    l1_weight: float = 1e-4

    # L2 regularization (Ridge / Weight Decay)
    use_l2: bool = False
    l2_weight: float = 1e-4

    def has_any_prior(self) -> bool:
        """Check if any prior is enabled."""
        return self.use_ard or self.use_entropy or self.use_smoothness or self.use_l1 or self.use_l2

    def to_dict(self) -> dict[str, t.Any]:
        """Convert PriorConfig to a dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, t.Any]) -> PriorConfig:
        """Create PriorConfig from a dictionary."""
        return cls(**data)
