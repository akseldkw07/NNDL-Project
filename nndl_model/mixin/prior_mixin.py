# pmml_project/mixin/prior_mixin.py
from __future__ import annotations

import torch
import torch.nn as nn

from ..priors import (
    ard_loss,
    categorical_gnm_entropy,
    compute_horseshoe_loss,
    compute_l1_loss,
    compute_l2_loss,
    smoothness_prior_hutchinson,
)
from .abc_nn import ABCNN
from .config import PriorConfig


class PriorMixin(ABCNN, nn.Module):
    """Mixin to add prior regularization to model training."""

    def set_prior(self, prior_config: PriorConfig) -> None:
        """Set the prior configuration."""
        self.prior_config = prior_config
        if prior_config.use_l2:
            curr_decay = self.optimizer.defaults.get("weight_decay", 0.0)
            decay_msg = f"weight decay: {curr_decay}!!" if curr_decay > 0.0 else "No weight decay set in optimizer."
            self.logger.warning(
                f"L2 prior enabled; Coupled with optimizer weight decay "
                f"may lead to double regularization. CURRENTLY: {decay_msg}"
            )

    def compute_prior_loss(
        self, inputs: torch.Tensor | None = None, outputs: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute total prior loss from all enabled priors.

        Args:
            inputs: Input tensor (required for smoothness prior)
            outputs: Model outputs/logits (required for entropy prior)

        Returns:
            Total prior loss as a scalar tensor
        """
        prior_loss = torch.tensor(0.0, device=self.device)

        if not self.prior_config.has_any_prior():
            return prior_loss

        # ARD loss (from model structure)
        if self.prior_config.use_ard:
            ard = ard_loss(self.model)
            prior_loss = prior_loss + self.prior_config.ard_weight * ard

        if self.prior_config.use_horseshoe:
            horseshoe = compute_horseshoe_loss(
                self.model, tau=self.prior_config.horseshoe_tau, epsilon=self.prior_config.horseshoe_epsilon
            )
            prior_loss = prior_loss + horseshoe
        # Entropy regularization (encourages uncertainty)
        if self.prior_config.use_entropy and outputs is not None:
            entropy = categorical_gnm_entropy(outputs)
            prior_loss = prior_loss + self.prior_config.entropy_weight * entropy

        # Smoothness prior (penalizes rapid changes)
        if self.prior_config.use_smoothness and inputs is not None:
            smoothness = smoothness_prior_hutchinson(
                self.model,
                inputs,
                lam=self.prior_config.smoothness_lambda,
                n_samples=self.prior_config.smoothness_samples,
            )
            prior_loss = prior_loss + smoothness

        # L1 regularization (encourages sparsity)
        if self.prior_config.use_l1:
            l1 = compute_l1_loss(self.model)
            prior_loss = prior_loss + self.prior_config.l1_weight * l1

        # L2 regularization (weight decay)
        if self.prior_config.use_l2:
            l2 = compute_l2_loss(self.model)
            prior_loss = prior_loss + self.prior_config.l2_weight * l2

        return prior_loss
