from __future__ import annotations

import torch
import torch.nn as nn
import tqdm
from torch import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from tqdm.contrib.logging import logging_redirect_tqdm

from .abc_nn import ABCNN
from .batch_loader import XTYPE, YTYPE
from .config import TrainingCallback, TrainingConfig, ret_updated_config


class SingleVariateMixin(ABCNN, nn.Module):
    early_stopping_triggered: bool = False

    def _train_step(self, inputs: torch.Tensor, labels: torch.Tensor, use_amp: bool = False) -> torch.Tensor:
        """
        Perform a single training step. Override for custom training logic.

        Args:
            inputs: Input tensor
            labels: Target tensor
            scaler: GradScaler for mixed precision (if enabled)
            use_amp: Whether to use automatic mixed precision

        Returns:
            Loss tensor
        """
        with autocast(device_type=self.device, enabled=use_amp):
            outputs = self(inputs)
            loss = self.get_loss(outputs, labels, inputs=inputs)
        return loss

    def train_model(
        self,
        train_loader: DataLoader | tuple[XTYPE, YTYPE],
        val_loader: DataLoader | tuple[XTYPE, YTYPE],
        num_epochs: int | None = None,
        batch_size: int | None = None,
        config_: TrainingConfig | None = None,
        callbacks: list[TrainingCallback] | None = None,
    ):
        """
        Train the model with enhanced flexibility.

        Args:
            train_loader: Training data (DataLoader or tuple of tensors)
            val_loader: Validation data (DataLoader or tuple of tensors)
            epochs: Number of epochs
            batch_size: Batch size
            config: TrainingConfig object with advanced options
            callbacks: Additional callbacks (merged with config.callbacks)
        """
        # Post_init check + config update
        if not self._post_init_done:
            raise RuntimeError("post_init must be called before training the model.")

        updated = ret_updated_config(
            self.training_config, config_ or self.training_config, num_epochs=num_epochs, batch_size=batch_size
        )
        if self.training_config != updated:
            with logging_redirect_tqdm([self.logger]):
                diffs = self.training_config.get_diff(updated)
                self.logger.info(f"Training config changes: {diffs}")
            self.training_config = updated
        cfg = self.training_config

        # Merge callbacks
        all_callbacks = cfg.callbacks.copy()
        if callbacks:
            all_callbacks.extend(callbacks)

        device = self.device
        epochs_no_improve = 0
        batch_size = cfg.batch_size

        train_loader = self._to_dataloader(train_loader, batch_size=batch_size)
        val_loader = self._to_dataloader(val_loader, batch_size=batch_size)

        # Setup mixed precision
        scaler = GradScaler() if cfg.use_amp and device == "cuda" else None

        # Callback: on_train_begin
        for cb in all_callbacks:
            cb.on_train_begin(self, logs={"config": cfg})  # TODO log initial eval to wandb

        global_step = 0

        # Training loop
        epoch_iterator = tqdm.tqdm(range(cfg.num_epochs)) if cfg.show_progress else range(cfg.num_epochs)

        for epoch in epoch_iterator:
            # Callback: on_epoch_begin
            for cb in all_callbacks:
                cb.on_epoch_begin(epoch, self)

            self.train()
            running_loss = 0.0
            total = 0
            self.optimizer.zero_grad()

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                # Callback: on_batch_begin
                for cb in all_callbacks:
                    cb.on_batch_begin(global_step, self)

                inputs: torch.Tensor = inputs.to(device)
                labels: torch.Tensor = labels.to(device)

                # Forward pass and loss computation
                loss = self._train_step(inputs, labels, cfg.use_amp)

                # Normalize loss for gradient accumulation
                loss = loss / cfg.gradient_accumulation_steps

                # Backward pass
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Callback: on_batch_end (after backward, before optimizer step)
                for cb in all_callbacks:
                    cb.on_batch_end(global_step, self, logs={"after_backward": True})

                # Optimizer step (with gradient accumulation)
                if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                    if scaler:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                # Track loss
                running_loss += loss.item() * inputs.size(0) * cfg.gradient_accumulation_steps
                total += inputs.size(0)
                global_step += 1

                # Batch-level logging
                if cfg.wandb_plot_every_n_batches and batch_idx % cfg.wandb_plot_every_n_batches == 0:
                    batch_loss = loss.item() * cfg.gradient_accumulation_steps
                    running_batch_loss = running_loss / total
                    self._log_wandb(
                        {"batch_loss": batch_loss, "running_batch_loss": running_batch_loss, "global_step": global_step}
                    )

                # Batch-level evaluation
                if cfg.eval_every_n_steps and global_step % cfg.eval_every_n_steps == 0:
                    eval_results = self.evaluate(val_loader)
                    self._log_wandb({"global_step": global_step, **eval_results})

            # Epoch-level metrics
            epoch_loss = running_loss / total

            # Epoch-level evaluation
            eval_results = {}
            if (epoch + 1) % cfg.eval_every_n_epochs == 0:
                eval_results = self.evaluate(val_loader)
                self._log_wandb({"train_loss": epoch_loss, "epoch": epoch, **eval_results})

                # Early stopping: check improvement
                improvements = self._improved(eval_results)
                epochs_no_improve = self._on_improvement(improvements, eval_results, epochs_no_improve)

            self.eval_best["epochs_trained"] += 1

            # Callback: on_epoch_end
            for cb in all_callbacks:
                cb.on_epoch_end(
                    epoch,
                    self,
                    logs={
                        "train_loss": epoch_loss,
                        **eval_results,
                        "epochs_no_improve": epochs_no_improve,
                    },
                )

            # Early stopping check
            early_stopping = self._patience_reached(epochs_no_improve)
            if early_stopping:
                self.early_stopping_triggered = True
                if cfg.show_progress:
                    tqdm.tqdm.write(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Callback: on_train_end
        for cb in all_callbacks:
            cb.on_train_end(self, logs={"total_epochs": self.eval_best["epochs_trained"]})
