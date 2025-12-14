import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from constants import *
from models import accuracy_from_logits, sub_probs_to_super_probs
from torch.utils.data import DataLoader


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    mode: str,
    sub_to_super: dict,
    num_super: int,
    alpha_kl=0.1,
    temperature=1.0,
):
    model.eval()
    running_loss = 0.0
    super_correct = sub_correct = 0
    super_total = sub_total = 0

    for images, super_labels, sub_labels in loader:
        images = images.to(DEVICE)
        super_labels = super_labels.to(DEVICE)
        sub_labels = sub_labels.to(DEVICE)
        loss = torch.Tensor([0.0]).to(DEVICE)

        if mode == "two_heads_kl":
            super_logits, sub_logits = model(images)
            loss_super = criterion(super_logits, super_labels)
            loss_sub = criterion(sub_logits, sub_labels)

            with torch.no_grad():
                super_probs = F.softmax(super_logits / temperature, dim=1)
            sub_probs = F.softmax(sub_logits / temperature, dim=1)
            agg_super_probs = sub_probs_to_super_probs(sub_probs, sub_to_super, num_super)

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

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    metrics = {"val_loss": avg_loss}
    if super_total > 0:
        metrics["val_acc_super"] = super_correct / super_total
    if sub_total > 0:
        metrics["val_acc_sub"] = sub_correct / sub_total

    return metrics


# Analysis and Visualize: for Novel Subclass fine tuning
# Determines optimal threshold value from training data (except held out) vs. held out (pseudo-novel) subclass data


@torch.no_grad()
def collect_max_probs_sub(model, loader, mode="two_heads"):
    """
    Collect max softmax probabilities from the subclass head.

    mode:
      - "two_heads": model(images) -> (super_logits, sub_logits)
      - "sub_single_head": model(images) -> sub_logits
    """
    model.eval()
    probs = []

    for batch in loader:
        images = batch[0].to(DEVICE)

        if mode == "two_heads":
            _, sub_logits = model(images)
        elif mode == "sub_single_head":
            sub_logits = model(images)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Softmax over subclasses, then take max per sample
        p = F.softmax(sub_logits, dim=1).max(dim=1).values
        probs.extend(p.cpu().numpy().tolist())

    return np.array(probs)


@torch.no_grad()
def analyze_tau_sub(model, pseudo_novel_loader: DataLoader | None, val_loader: DataLoader, mode="two_heads"):
    """
    Compare max subclass probabilities on:
      - seen validation (val_loader)
      - pseudo-novel validation (pseudo_novel_loader)
    and suggest TAU_SUB candidates.
    """

    if pseudo_novel_loader is None:
        print(
            "pseudo_novel_loader is None. " "Set USE_PSEUDO_NOVEL = True before building loaders to use this analysis."
        )
        return

    # Collect max probs
    seen_probs = collect_max_probs_sub(model, val_loader, mode=mode)
    pseudo_probs = collect_max_probs_sub(model, pseudo_novel_loader, mode=mode)

    # Summary stats
    def summarize(name, arr):
        print(f"\n{name} subclass max-prob stats:")
        print(f"  count = {len(arr)}")
        print(f"  mean  = {arr.mean():.3f}")
        print(f"  std   = {arr.std():.3f}")
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pvals = {p: float(np.percentile(arr, p)) for p in percentiles}
        print("  percentiles:")
        for p in percentiles:
            print(f"    p{p:>2}: {pvals[p]:.3f}")
        return pvals

    seen_p = summarize("Seen", seen_probs)
    summarize("Pseudo-novel", pseudo_probs)

    # Simple candidate thresholds
    # 1) 10th percentile of seen (reject only the lowest-confidence seen examples)
    tau_candidate_1 = seen_p[10]

    # 2) Midpoint between mean(seen) and mean(pseudo)
    tau_candidate_2 = 0.5 * (seen_probs.mean() + pseudo_probs.mean())

    print("\nSuggested TAU_SUB candidates:")
    print(f"  tau_sub ≈ 10th percentile of seen: {tau_candidate_1:.3f}")
    print(f"  tau_sub ≈ mean(seen + pseudo)/2:  {tau_candidate_2:.3f}")
    print("\nYou can start with one of these for TAU_SUB and adjust based on leaderboard/behavior.")

    # Histogram visualization
    plt.figure(figsize=(8, 5))
    plt.hist(seen_probs, bins=30, alpha=0.5, label="Seen subclasses")
    plt.hist(pseudo_probs, bins=30, alpha=0.5, label="Pseudo-novel subclasses")
    plt.axvline(tau_candidate_1, linestyle="--", label=f"p10 seen ~ {tau_candidate_1:.2f}")
    plt.axvline(tau_candidate_2, linestyle=":", label=f"mean midpoint ~ {tau_candidate_2:.2f}")
    plt.xlabel("Max softmax probability (subclass head)")
    plt.ylabel("Count")
    plt.title("Subclass max-prob: seen vs pseudo-novel")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optional: log histograms and candidates to W&B

    wandb.log(
        {
            "sub_seen_maxprob_hist": wandb.Histogram(seen_probs),  # type:ignore[call-arg]
            "sub_pseudo_maxprob_hist": wandb.Histogram(pseudo_probs),  # type:ignore[call-arg]
            "tau_sub_candidate_p10_seen": tau_candidate_1,
            "tau_sub_candidate_mean_midpoint": tau_candidate_2,
        }
    )
    print("Logged histograms and tau_sub candidates to Weights & Biases.")


# Analysis and Visualize: for Novel Superclass fine tuning
# Determines the optimal threshold from provided training data vs. Novel Super data (more images)


@torch.no_grad()
def collect_max_probs_super(model, loader, mode="two_heads"):
    """
    Collect max softmax probabilities from the superclass head.

    mode:
      - "two_heads": model(images) -> (super_logits, sub_logits)
      - "super_single_head": model(images) -> super_logits
    """
    model.eval()
    probs = []

    for batch in loader:
        images = batch[0].to(DEVICE)

        if mode == "two_heads":
            super_logits, _ = model(images)
        elif mode == "super_single_head":
            super_logits = model(images)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        p = F.softmax(super_logits, dim=1).max(dim=1).values
        probs.extend(p.cpu().numpy().tolist())

    return np.array(probs)


@torch.no_grad()
def analyze_tau_super(model, val_loader: DataLoader, NOVEL_SUPER_IDX: int, mode="two_heads"):
    """
    Analyze max superclass probabilities on validation set, split into:
      - seen superclasses (super != NOVEL_SUPER_IDX)
      - novel superclasses (super == NOVEL_SUPER_IDX)

    This assumes:
      - val_loader batches look like (images, super_labels, sub_labels, ...)
      - NOVEL_SUPER_IDX is defined (e.g. 3)
    """

    model.eval()

    seen_probs = []
    novel_probs = []

    for batch in val_loader:
        images = batch[0].to(DEVICE)
        super_labels = batch[1].to(DEVICE)  # assumes (images, super, sub, ...)

        # Forward
        if mode == "two_heads":
            super_logits, _ = model(images)
        elif mode == "super_single_head":
            super_logits = model(images)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        probs = F.softmax(super_logits, dim=1)  # (B, num_super)
        max_probs, _ = probs.max(dim=1)  # (B,)

        novel_mask = super_labels == NOVEL_SUPER_IDX
        seen_mask = ~novel_mask

        if seen_mask.any():
            seen_probs.extend(max_probs[seen_mask].detach().cpu().numpy().tolist())
        if novel_mask.any():
            novel_probs.extend(max_probs[novel_mask].detach().cpu().numpy().tolist())

    seen_probs = np.array(seen_probs)
    novel_probs = np.array(novel_probs)

    def summarize(name, arr):
        print(f"\n{name} superclass max-prob stats:")
        print(f"  count = {len(arr)}")
        print(f"  mean  = {arr.mean():.3f}")
        print(f"  std   = {arr.std():.3f}")
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pvals = {p: float(np.percentile(arr, p)) for p in percentiles}
        print("  percentiles:")
        for p in percentiles:
            print(f"    p{p:>2}: {pvals[p]:.3f}")
        return pvals

    if len(seen_probs) == 0:
        print("No seen-super examples found in val_loader (super != NOVEL_SUPER_IDX).")
        return
    seen_p = summarize("Seen superclasses (0/1/2)", seen_probs)

    if len(novel_probs) == 0:
        print("\nNo novel-super examples (super == NOVEL_SUPER_IDX) in val_loader yet.")
        novel_p = None
    else:
        novel_p = summarize("Novel superclasses (== NOVEL_SUPER_IDX)", novel_probs)

    # ---- Candidate thresholds ----
    # Start from seen distribution:
    tau_p10_seen = seen_p[10]
    tau_p5_seen = seen_p[5]
    tau_mean_minus_std = max(0.0, min(1.0, seen_probs.mean() - seen_probs.std()))

    # If we have novel-super stats, try to place tau between seen & novel means/percentiles
    if novel_p is not None:
        # Midpoint between seen mean and novel mean
        tau_mean_mid = 0.5 * (seen_probs.mean() + novel_probs.mean())
        print("\nSuggested TAU_SUPER candidates (using seen + novel):")
        print(f"  tau_super ≈ 10th percentile of seen:       {tau_p10_seen:.3f}")
        print(f"  tau_super ≈ 5th percentile of seen:        {tau_p5_seen:.3f}")
        print(f"  tau_super ≈ mean(seen) - std(seen):        {tau_mean_minus_std:.3f}")
        print(f"  tau_super ≈ mean(seen & novel) midpoint:   {tau_mean_mid:.3f}")
        tau_candidates = [tau_p10_seen, tau_p5_seen, tau_mean_minus_std, tau_mean_mid]
    else:
        print("\nSuggested TAU_SUPER candidates (seen only):")
        print(f"  tau_super ≈ 10th percentile of seen:       {tau_p10_seen:.3f}")
        print(f"  tau_super ≈ 5th percentile of seen:        {tau_p5_seen:.3f}")
        print(f"  tau_super ≈ mean(seen) - std(seen):        {tau_mean_minus_std:.3f}")
        tau_candidates = [tau_p10_seen, tau_p5_seen, tau_mean_minus_std]

    # ---- Histogram plot ----
    plt.figure(figsize=(8, 5))
    plt.hist(seen_probs, bins=30, alpha=0.6, label="Seen superclasses (0/1/2)")
    if len(novel_probs) > 0:
        plt.hist(novel_probs, bins=30, alpha=0.6, label="Novel superclasses (3)")

    # Draw candidate lines (use a couple of them for visual reference)
    for tau in tau_candidates[:3]:
        plt.axvline(tau, linestyle="--", alpha=0.7)

    plt.xlabel("Max softmax probability (superclass head)")
    plt.ylabel("Count")
    plt.title("Superclass max-prob: seen vs novel-super on validation")
    plt.legend()
    plt.tight_layout()
    plt.show()

    log_dict: dict[str, wandb.Histogram | float] = {
        "super_seen_maxprob_hist": wandb.Histogram(seen_probs),  # type:ignore[call-arg]
    }
    if len(novel_probs) > 0:
        log_dict["super_novel_maxprob_hist"] = wandb.Histogram(novel_probs)  # type:ignore[call-arg]
    # Also log first few tau candidates
    log_dict["tau_super_candidate_1"] = float(tau_candidates[0])
    if len(tau_candidates) > 1:
        log_dict["tau_super_candidate_2"] = float(tau_candidates[1])
    if len(tau_candidates) > 2:
        log_dict["tau_super_candidate_3"] = float(tau_candidates[2])
    wandb.log(log_dict)
    print("Logged superclass histograms and tau_super candidates to Weights & Biases.")


# Additional eval helper functions

# This calculates the following:
# 1. How often we correctly keep seen classes (low false "novel" rate)
# 2. How often we correctly identify CIFAR "novel super" samples (new data)


@torch.no_grad()
def evaluate_on_val_with_novelty(
    model, val_loader: DataLoader, mode="two_heads", tau_super=TAU_SUPER, tau_sub=TAU_SUB, name="val"
):
    """
    Evaluate a model on a loader with novel thresholds applied.

    For subclasses, we ONLY care about:
      - overall subclass acc
      - seen-subclass acc / false-novel rate

    We DO NOT report "novel-subclass accuracy" here because there are no
    ground-truth novel-sub labels in the original validation data.
    """

    model.eval()

    total = 0
    super_correct = 0
    sub_correct = 0

    seen_super_correct = seen_super_total = 0
    novel_super_correct = novel_super_total = 0

    seen_sub_correct = seen_sub_total = 0
    novel_sub_correct = novel_sub_total = 0  # still counted but not reported

    for batch in val_loader:
        images = batch[0].to(DEVICE)
        super_true = batch[1].to(DEVICE)
        sub_true = batch[2].to(DEVICE)

        if mode == "two_heads":
            super_logits, sub_logits = model(images)
        elif mode == "super_single_head":
            super_logits = model(images)
            sub_logits = None
        elif mode == "sub_single_head":
            super_logits = None
            sub_logits = model(images)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        batch_size = images.size(0)
        total += batch_size

        # --- Superclass ---
        if super_logits is not None:
            super_probs = F.softmax(super_logits, dim=1)
            max_super_probs, super_idx = super_probs.max(dim=1)

            super_pred = super_idx.clone()
            novel_mask_super = max_super_probs < tau_super
            super_pred[novel_mask_super] = NOVEL_SUPER_IDX

            match_super = super_pred == super_true
            super_correct += match_super.sum().item()

            seen_mask_super = super_true != NOVEL_SUPER_IDX
            novel_mask_super_label = super_true == NOVEL_SUPER_IDX

            if seen_mask_super.any():
                seen_super_correct += match_super[seen_mask_super].sum().item()
                seen_super_total += seen_mask_super.sum().item()
            if novel_mask_super_label.any():
                novel_super_correct += match_super[novel_mask_super_label].sum().item()
                novel_super_total += novel_mask_super_label.sum().item()

        # --- Subclass ---
        if sub_logits is not None:
            sub_probs = F.softmax(sub_logits, dim=1)
            max_sub_probs, sub_idx = sub_probs.max(dim=1)

            sub_pred = sub_idx.clone()
            novel_mask_sub = max_sub_probs < tau_sub
            sub_pred[novel_mask_sub] = NOVEL_SUB_IDX

            match_sub = sub_pred == sub_true
            sub_correct += match_sub.sum().item()

            seen_mask_sub = sub_true != NOVEL_SUB_IDX
            novel_mask_sub_label = sub_true == NOVEL_SUB_IDX

            if seen_mask_sub.any():
                seen_sub_correct += match_sub[seen_mask_sub].sum().item()
                seen_sub_total += seen_mask_sub.sum().item()
            if novel_mask_sub_label.any():
                # counted but not reported; there shouldn't be any in original val
                novel_sub_correct += match_sub[novel_mask_sub_label].sum().item()
                novel_sub_total += novel_mask_sub_label.sum().item()

    # ----- Compute metrics dict -----
    metrics = {}

    if total > 0:
        if super_correct > 0 or seen_super_total + novel_super_total > 0:
            metrics["overall_super_acc"] = super_correct / total
        if sub_correct > 0 or seen_sub_total + novel_sub_total > 0:
            metrics["overall_sub_acc"] = sub_correct / total

    if seen_super_total > 0:
        metrics["seen_super_acc"] = seen_super_correct / seen_super_total
        metrics["seen_super_false_novel"] = 1.0 - metrics["seen_super_acc"]
    if novel_super_total > 0:
        metrics["novel_super_acc"] = novel_super_correct / novel_super_total

    if seen_sub_total > 0:
        metrics["seen_sub_acc"] = seen_sub_correct / seen_sub_total
        metrics["seen_sub_false_novel"] = 1.0 - metrics["seen_sub_acc"]
    # NOTE: we deliberately do NOT add novel-sub metrics to `metrics`,
    # because they are not meaningful for this dataset.

    # ----- Print summary -----
    print(f"\n=== Evaluation on {name} ===")
    if "overall_super_acc" in metrics:
        print(f"Overall superclass acc: {metrics['overall_super_acc']:.4f}")
    if "overall_sub_acc" in metrics:
        print(f"Overall subclass acc:   {metrics['overall_sub_acc']:.4f}")

    if "seen_super_acc" in metrics:
        print(f"Seen superclass acc (true super != novel):   {metrics['seen_super_acc']:.4f}")
        print(f"Seen superclass false-novel rate:            {metrics['seen_super_false_novel']:.4f}")
    if "novel_super_acc" in metrics:
        print(f"Novel superclass acc (true super == novel):  {metrics['novel_super_acc']:.4f}")

    if "seen_sub_acc" in metrics:
        print(f"Seen subclass acc (true sub != novel):       {metrics['seen_sub_acc']:.4f}")
        print(f"Seen subclass false-novel rate:              {metrics['seen_sub_false_novel']:.4f}")

    # ----- Optional: log to W&B (only real metrics) -----
    log_dict = {}
    for k, v in metrics.items():
        log_dict[f"{name}_{k}"] = v
    if log_dict:
        wandb.log(log_dict)

    return metrics


# This enables us to evaluate how often we correctly mark held-out subclasses as novel (proxy for leaderboard performance on subclasses)
@torch.no_grad()
def evaluate_pseudo_novel_sub_with_novelty(
    model, pseudo_novel_loader: DataLoader | None = None, mode="two_heads", tau_sub=TAU_SUB, name="pseudo_novel_sub"
):
    """
    Evaluate how well the model flags held-out subclasses as novel.

    Assumes:
      - loader yields only *held-out* subclasses (true unseen subclasses)
      - true labels are NOT NOVEL_SUB_IDX, but we *want* the model to predict NOVEL_SUB_IDX.
    """
    if pseudo_novel_loader is None:
        print("No pseudo_novel_loader available.")
        return {}

    model.eval()

    total = 0
    predicted_novel = 0
    predicted_seen = 0

    for batch in pseudo_novel_loader:
        images = batch[0].to(DEVICE)
        # we don't actually need the labels here for correctness, only for counting

        if mode == "two_heads":
            _, sub_logits = model(images)
        elif mode == "sub_single_head":
            sub_logits = model(images)
        else:
            raise ValueError(f"Unknown mode for pseudo-novel eval: {mode}")

        sub_probs = F.softmax(sub_logits, dim=1)
        max_sub_probs, sub_idx = sub_probs.max(dim=1)

        sub_pred = sub_idx.clone()
        novel_mask_sub = max_sub_probs < tau_sub
        sub_pred[novel_mask_sub] = NOVEL_SUB_IDX

        batch_size = images.size(0)
        total += batch_size

        predicted_novel += (sub_pred == NOVEL_SUB_IDX).sum().item()
        predicted_seen += (sub_pred != NOVEL_SUB_IDX).sum().item()

    metrics = {}
    if total > 0:
        metrics["pseudo_novel_sub_novel_rate"] = predicted_novel / total
        metrics["pseudo_novel_sub_false_seen"] = predicted_seen / total

    print(f"\n=== Evaluation on {name} (held-out subclasses) ===")
    if total > 0:
        print(f"Fraction flagged as novel (good): {metrics['pseudo_novel_sub_novel_rate']:.4f}")
        print(f"Fraction mapped to seen subclasses (bad): {metrics['pseudo_novel_sub_false_seen']:.4f}")

    if metrics:
        log_dict = {f"{name}_{k}": v for k, v in metrics.items()}
        wandb.log(log_dict)

    return metrics


# Dashboard for our evals so we can determine how model will likely perform on leaderboard evaluation


@torch.no_grad()
def novelty_dashboard(
    model,
    val_loader: DataLoader,
    pseudo_novel_loader: DataLoader | None = None,
    mode="two_heads",
    tau_super=TAU_SUPER,
    tau_sub=TAU_SUB,
    include_pseudo=True,
):
    """
    Run thresholded evals and show a compact table of key metrics.

    Subclass side focuses on:
      - seen subclasses (false-novel rate)
      - held-out pseudo-novel subclasses (how often marked as novel)
    """
    rows = []

    # --- Main val_loader stats (seen + CIFAR-novel) ---
    val_metrics = evaluate_on_val_with_novelty(
        model,
        mode=mode,
        tau_super=tau_super,
        tau_sub=tau_sub,
        val_loader=val_loader,
        name="val",
    )

    # Superclass (val)
    if "seen_super_acc" in val_metrics:
        rows.append(
            {
                "Split": "val",
                "Head": "super",
                "Metric": "Seen superclass accuracy",
                "Meaning": "Correctly keep seen superclasses as seen",
                "Value": float(val_metrics["seen_super_acc"]),
            }
        )
        if "seen_super_false_novel" in val_metrics:
            rows.append(
                {
                    "Split": "val",
                    "Head": "super",
                    "Metric": "Seen superclass false-novel rate",
                    "Meaning": "Seen superclasses incorrectly flipped to novel",
                    "Value": float(val_metrics["seen_super_false_novel"]),
                }
            )

    if "novel_super_acc" in val_metrics:
        rows.append(
            {
                "Split": "val",
                "Head": "super",
                "Metric": "Novel superclass accuracy (CIFAR)",
                "Meaning": "CIFAR novel-super samples correctly predicted as novel",
                "Value": float(val_metrics["novel_super_acc"]),
            }
        )

    # Subclass (val) — ONLY seen-subclass metrics
    if "seen_sub_acc" in val_metrics:
        rows.append(
            {
                "Split": "val",
                "Head": "sub",
                "Metric": "Seen subclass accuracy",
                "Meaning": "Correctly keep seen subclasses as seen",
                "Value": float(val_metrics["seen_sub_acc"]),
            }
        )
        if "seen_sub_false_novel" in val_metrics:
            rows.append(
                {
                    "Split": "val",
                    "Head": "sub",
                    "Metric": "Seen subclass false-novel rate",
                    "Meaning": "Seen subclasses incorrectly flipped to novel",
                    "Value": float(val_metrics["seen_sub_false_novel"]),
                }
            )
    # NOTE: we deliberately do NOT add a "novel subclass accuracy" row here.

    # --- Pseudo-novel subclass stats (held-out subclasses) ---
    pseudo_metrics = {}
    if include_pseudo and pseudo_novel_loader is not None and mode in ("two_heads", "sub_single_head"):
        pseudo_metrics = evaluate_pseudo_novel_sub_with_novelty(
            model,
            mode=mode,
            tau_sub=tau_sub,
            pseudo_novel_loader=pseudo_novel_loader,
            name="pseudo_novel_sub",
        )

        if "pseudo_novel_sub_novel_rate" in pseudo_metrics:
            rows.append(
                {
                    "Split": "pseudo_novel",
                    "Head": "sub",
                    "Metric": "Pseudo-novel marked as novel",
                    "Meaning": "Held-out subclasses correctly flagged as novel",
                    "Value": float(pseudo_metrics["pseudo_novel_sub_novel_rate"]),
                }
            )
        if "pseudo_novel_sub_false_seen" in pseudo_metrics:
            rows.append(
                {
                    "Split": "pseudo_novel",
                    "Head": "sub",
                    "Metric": "Pseudo-novel mapped to seen",
                    "Meaning": "Held-out subclasses wrongly mapped to seen subclass",
                    "Value": float(pseudo_metrics["pseudo_novel_sub_false_seen"]),
                }
            )

    # --- Config / settings summary (for *printing* only) ---
    config_rows = [
        {
            "Split": "config",
            "Head": "-",
            "Metric": "BACKBONE",
            "Meaning": "Feature extractor (e.g. resnet18 / resnet50)",
            "Value": BACKBONE,
        },
        {
            "Split": "config",
            "Head": "-",
            "Metric": "TAU_SUPER",
            "Meaning": "Novelty threshold for superclass head",
            "Value": str(tau_super),
        },
        {
            "Split": "config",
            "Head": "-",
            "Metric": "TAU_SUB",
            "Meaning": "Novelty threshold for subclass head",
            "Value": str(tau_sub),
        },
        {
            "Split": "config",
            "Head": "-",
            "Metric": "CIFAR_NOVEL_MODE",
            "Meaning": "Extra novel-super CIFAR data mode",
            "Value": str(CIFAR_NOVEL_MODE),
        },
        {
            "Split": "config",
            "Head": "-",
            "Metric": "FINE_TUNE_MODE",
            "Meaning": "Backbone training mode (full vs frozen)",
            "Value": str(FINE_TUNE_MODE),
        },
        {
            "Split": "config",
            "Head": "-",
            "Metric": "APPROACH",
            "Meaning": "Model architecture (two_heads vs two_models)",
            "Value": str(APPROACH),
        },
        {
            "Split": "config",
            "Head": "-",
            "Metric": "USE_PSEUDO_NOVEL",
            "Meaning": "Using held-out subclasses for pseudo-novel eval",
            "Value": str(bool(USE_PSEUDO_NOVEL)),
        },
        {
            "Split": "config",
            "Head": "-",
            "Metric": "DATA_AUGMENT",
            "Meaning": "Whether data augmentation is enabled for training",
            "Value": str(bool(DATA_AUGMENT)),
        },
        {
            "Split": "config",
            "Head": "-",
            "Metric": "SUB_HEAD_TYPE",
            "Meaning": "Type of subclass head (linear or cosine)",
            "Value": str(SUB_HEAD_TYPE),
        },
        {
            "Split": "config",
            "Head": "-",
            "Metric": "SUPER_HEAD_TYPE",
            "Meaning": "Type of superclass head (linear or cosine)",
            "Value": str(SUPER_HEAD_TYPE),
        },
    ]

    rows.extend(config_rows)

    if not rows:
        print("No metrics to display in dashboard.")
        return None

    dashboard_df = pd.DataFrame(rows)
    dashboard_df = dashboard_df.sort_values(by=["Split", "Head", "Metric"]).reset_index(drop=True)

    print("\n==== Novelty Dashboard ====")
    print(dashboard_df)

    # --- Log to W&B: metrics only, numeric Value column ---
    wandb_table_df = dashboard_df.copy()

    def _to_str(v):
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)

    wandb_table_df["Split"] = wandb_table_df["Split"].astype(str)
    wandb_table_df["Head"] = wandb_table_df["Head"].astype(str)
    wandb_table_df["Metric"] = wandb_table_df["Metric"].astype(str)
    wandb_table_df["Meaning"] = wandb_table_df["Meaning"].astype(str)
    wandb_table_df["Value"] = wandb_table_df["Value"].apply(_to_str)

    wandb.log({"novelty_dashboard": wandb.Table(dataframe=wandb_table_df)})

    return dashboard_df
