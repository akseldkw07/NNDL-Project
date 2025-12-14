import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from constants import DEVICE as device
from models import accuracy_from_logits, sub_probs_to_super_probs

from constants import *

# functions for training

# we pass a flag mode to indicate which model using either:
# "two_heads_kl", "single_head_super" or "single_head"sub"


def train_one_epoch(
    model, optimizer, loader, criterion, mode, sub_to_super=None, num_super=None, alpha_kl=0.1, temperature=1.0
):
    model.train()
    running_loss = 0.0
    super_correct = sub_correct = 0
    super_total = sub_total = 0

    for images, super_labels, sub_labels in loader:
        images = images.to(device)
        super_labels = super_labels.to(device)
        sub_labels = sub_labels.to(device)

        optimizer.zero_grad()

        if mode == "two_heads_kl":
            super_logits, sub_logits = model(images)
            # CE losses
            loss_super = criterion(super_logits, super_labels)
            loss_sub = criterion(sub_logits, sub_labels)

            # KL term between super head and aggregated subclass head
            with torch.no_grad():
                # target: super_probs
                super_probs = F.softmax(super_logits / temperature, dim=1)
            sub_probs = F.softmax(sub_logits / temperature, dim=1)
            agg_super_probs = sub_probs_to_super_probs(sub_probs, sub_to_super, num_super)

            # KL(super || agg_super) = sum p * (log p - log q)
            # using KLDivLoss with log_softmax input and probs target:
            kl_loss = F.kl_div(input=torch.log(agg_super_probs + 1e-8), target=super_probs, reduction="batchmean")

            loss = loss_super + loss_sub + alpha_kl * kl_loss

            sc, st = accuracy_from_logits(super_logits, super_labels)
            suc, sut = accuracy_from_logits(sub_logits, sub_labels)
            super_correct += sc
            super_total += st
            sub_correct += suc
            sub_total += sut

        elif mode in ("single_head_super", "single_head_sub"):
            logits = model(images)
            if mode == "single_head_super":
                loss = criterion(logits, super_labels)
                sc, st = accuracy_from_logits(logits, super_labels)
                super_correct += sc
                super_total += st
            else:
                loss = criterion(logits, sub_labels)
                suc, sut = accuracy_from_logits(logits, sub_labels)
                sub_correct += suc
                sub_total += sut
        else:
            raise ValueError(f"Unknown mode {mode}")

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    metrics = {"loss": avg_loss}
    if super_total > 0:
        metrics["acc_super"] = super_correct / super_total
    if sub_total > 0:
        metrics["acc_sub"] = sub_correct / sub_total

    return metrics


def split_seen_vs_novel_super(df, val_split, novel_super_idx, rng_seed=42):
    """
    Split dataframe into train/val, ensuring that both seen-super (0/1/2)
    and novel-super (== novel_super_idx) appear in both splits if present.
    """
    rng = np.random.default_rng(rng_seed)

    df_novel_super = df[df["superclass_index"] == novel_super_idx]
    df_seen_super = df[df["superclass_index"] != novel_super_idx]

    print("  Seen-super samples:", len(df_seen_super))
    print("  Novel-super samples:", len(df_novel_super))

    # Split seen-super part
    if len(df_seen_super) > 0:
        val_seen_size = int(len(df_seen_super) * val_split)
        val_seen_indices = rng.choice(len(df_seen_super), size=val_seen_size, replace=False)
        val_seen_df = df_seen_super.iloc[val_seen_indices]
        train_seen_df = df_seen_super.drop(val_seen_df.index)
    else:
        val_seen_df = df_seen_super.iloc[0:0]
        train_seen_df = df_seen_super.iloc[0:0]

    # Split novel-super part (if any)
    if len(df_novel_super) > 0:
        val_novel_size = max(1, int(len(df_novel_super) * val_split))
        val_novel_indices = rng.choice(len(df_novel_super), size=val_novel_size, replace=False)
        val_novel_df = df_novel_super.iloc[val_novel_indices]
        train_novel_df = df_novel_super.drop(val_novel_df.index)
    else:
        val_novel_df = df_novel_super.iloc[0:0]
        train_novel_df = df_novel_super.iloc[0:0]

    # Combine splits and shuffle
    train_split_df = pd.concat([train_seen_df, train_novel_df], ignore_index=True)
    val_split_df = pd.concat([val_seen_df, val_novel_df], ignore_index=True)

    train_split_df = train_split_df.sample(frac=1.0, random_state=rng_seed).reset_index(drop=True)
    val_split_df = val_split_df.sample(frac=1.0, random_state=rng_seed + 1).reset_index(drop=True)

    print("  Final split sizes:")
    print("    train:", len(train_split_df))
    print("    val:  ", len(val_split_df))
    print("    train novel-super:", (train_split_df["superclass_index"] == novel_super_idx).sum())
    print("    val novel-super:  ", (val_split_df["superclass_index"] == novel_super_idx).sum())

    return train_split_df, val_split_df


# Helper to choose backbone (freeze or full) and choose optimizer parameter


def setup_backbone_training(model, fine_tune_mode="full", lr_full=1e-4, lr_head=1e-3):
    """
    Given a model with attributes:
        - model.backbone  (all feature extractor layers)
        - head layers (e.g. super/sub heads) as other modules,
    freeze or unfreeze the backbone and return an optimizer.

    Returns:
        optimizer, effective_lr
    """
    if fine_tune_mode == "full":
        # Everything trainable
        for p in model.parameters():
            p.requires_grad = True
        trainable_params = model.parameters()
        lr = lr_full
    elif fine_tune_mode == "frozen":
        # Freeze backbone, train only heads
        for p in model.backbone.parameters():
            p.requires_grad = False
        # Only parameters that still require grad will be optimized
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        lr = lr_head
        print(f"Freezing backbone; training {len(trainable_params)} parameter tensors in heads only.")
    else:
        raise ValueError(f"Unknown FINE_TUNE_MODE: {fine_tune_mode}")

    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=WEIGHT_DECAY)
    return optimizer
