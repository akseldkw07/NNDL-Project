import pandas as pd
import torch
import torch.nn.functional as F
from constants import *
from constants import DEVICE as device

# Inference in order to predict "novel" superclass / subclasses


@torch.no_grad()
def predict_test_two_heads(model, test_loader, tau_super=TAU_SUPER, tau_sub=TAU_SUB):
    """
    Inference for the two-heads model with novelty thresholds.

    model:       SharedBackboneTwoHeads(...)
    test_loader: DataLoader yielding (images, img_names)
    tau_super:   threshold for superclass novelty
    tau_sub:     threshold for subclass novelty
    """
    model.eval()
    images_list = []
    super_preds = []
    sub_preds = []

    for images, img_names in test_loader:
        images = images.to(device, non_blocking=True)

        # Forward pass once
        super_logits, sub_logits = model(images)  # (B, num_super), (B, num_sub)

        # --- Superclass predictions with novelty ---
        super_probs = F.softmax(super_logits, dim=1)  # (B, num_super)
        max_super_probs, super_idx = super_probs.max(dim=1)  # (B,)
        super_novel_mask = max_super_probs < tau_super
        super_idx = super_idx.clone()
        super_idx[super_novel_mask] = NOVEL_SUPER_IDX

        # --- Subclass predictions with novelty ---
        sub_probs = F.softmax(sub_logits, dim=1)  # (B, num_sub)
        max_sub_probs, sub_idx = sub_probs.max(dim=1)  # (B,)
        sub_novel_mask = max_sub_probs < tau_sub
        sub_idx = sub_idx.clone()
        sub_idx[sub_novel_mask] = NOVEL_SUB_IDX

        # Move to CPU as plain Python ints
        super_idx = super_idx.cpu().tolist()
        sub_idx = sub_idx.cpu().tolist()

        # img_names is a list of filenames (len = B)
        images_list.extend(img_names)
        super_preds.extend(super_idx)
        sub_preds.extend(sub_idx)

    df = pd.DataFrame({"image": images_list, "superclass_index": super_preds, "subclass_index": sub_preds})
    return df


@torch.no_grad()
def predict_test_two_models(model_super, model_sub, test_loader, tau_super=TAU_SUPER, tau_sub=TAU_SUB):
    """
    Inference for the two-model setup (separate super + sub models) with novelty thresholds.

    model_super: SingleHeadModel for superclass (num_classes = num_super)
    model_sub:   SingleHeadModel for subclass  (num_classes = num_sub)
    test_loader: DataLoader yielding (images, img_names)
    tau_super:   threshold for superclass novelty
    tau_sub:     threshold for subclass novelty
    """

    model_super.eval()
    model_sub.eval()

    images_list = []
    super_preds = []
    sub_preds = []

    for images, img_names in test_loader:
        images = images.to(device, non_blocking=True)

        # Forward passes
        super_logits = model_super(images)  # (B, num_super)
        sub_logits = model_sub(images)  # (B, num_sub)

        # --- Superclass predictions with novelty ---
        super_probs = F.softmax(super_logits, dim=1)  # (B, num_super)
        max_super_probs, super_idx = super_probs.max(dim=1)  # (B,)
        super_novel_mask = max_super_probs < tau_super
        super_idx = super_idx.clone()
        super_idx[super_novel_mask] = NOVEL_SUPER_IDX

        # --- Subclass predictions with novelty ---
        sub_probs = F.softmax(sub_logits, dim=1)  # (B, num_sub)
        max_sub_probs, sub_idx = sub_probs.max(dim=1)  # (B,)
        sub_novel_mask = max_sub_probs < tau_sub
        sub_idx = sub_idx.clone()
        sub_idx[sub_novel_mask] = NOVEL_SUB_IDX

        # Move indices to CPU as plain Python ints
        super_idx = super_idx.cpu().tolist()
        sub_idx = sub_idx.cpu().tolist()

        # img_names is a list of filenames (len = B)
        images_list.extend(img_names)
        super_preds.extend(super_idx)
        sub_preds.extend(sub_idx)

    df = pd.DataFrame(
        {
            "image": images_list,
            "superclass_index": super_preds,
            "subclass_index": sub_preds,
        }
    )
    return df
