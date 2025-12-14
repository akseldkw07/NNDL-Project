import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_l1_loss(model: nn.Module) -> torch.Tensor:
    """
    Compute the L1 regularization loss for all parameters in the model.

    Args:
        model: Neural network module

    Returns:
        Total L1 loss as a scalar tensor
    """
    l1_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        l1_loss = l1_loss + torch.sum(torch.abs(param))
    return l1_loss


def compute_l2_loss(model: nn.Module) -> torch.Tensor:
    """
    Compute the L2 regularization loss for all parameters in the model.

    Args:
        model: Neural network module

    Returns:
        Total L2 loss as a scalar tensor
    """
    l2_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        l2_loss = l2_loss + torch.sum(param**2)
    return l2_loss


def categorical_gnm_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute the entropy of categorical predictions from logits.

    Uses the generalized natural mean (GNM) formulation to calculate
    the entropy of the categorical distribution defined by the input logits.

    Args:
        logits: Tensor of shape (batch_size, num_classes) containing
                unnormalized log probabilities

    Returns:
        Mean entropy across the batch as a scalar tensor
    """
    a_eta = torch.logsumexp(logits, dim=1)
    probs = F.softmax(logits, dim=1)
    dot_prod = (logits * probs).sum(dim=1)
    entropy = -dot_prod + a_eta
    return torch.mean(entropy)


def smoothness_prior_hutchinson(model, x: torch.Tensor, lam: float, n_samples: int = 1):
    """
    Approximates the sum of squared second derivatives using Hutchinson's trick.

    Ngl, idk what this does fully, but the version that derives the full gradients
    is too memory intesive. Unclear if this is doing the right things tho
    """
    x = x.clone().detach().requires_grad_(True)
    B = x.shape[0]

    y = model(x).squeeze()  # shape [B]

    prior = 0.0
    for _ in range(n_samples):
        # Random vector v same shape as x
        v = torch.randint_like(x, low=0, high=2) * 2 - 1  # Rademacher ±1

        # First derivative
        grads = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]  # shape [B, C, H, W]

        # Hessian-vector product
        Hv = torch.autograd.grad(grads, x, grad_outputs=v, retain_graph=True, create_graph=False)[0]

        prior += (Hv**2).sum()

    prior = (lam / (2 * B * n_samples)) * prior
    return prior


def compute_horseshoe_loss(model: nn.Module, tau: float = 0.1, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Compute the horseshoe prior regularization loss for all parameters in the model.

    The horseshoe prior has the form:
        p(θ) = ∫ N(θ | 0, λ²τ²) C⁺(λ | 0, 1) dλ

    This leads to a log-prior (negative regularization loss):
        -log p(θ) ≈ log(1 + θ²/(τ²))  [using the half-Cauchy integral]

    The horseshoe prior provides strong shrinkage for small parameters while
    allowing large parameters to escape regularization (heavy tails).

    Args:
        model: Neural network module
        tau: Global shrinkage parameter (controls overall sparsity level)
        epsilon: Small constant for numerical stability

    Returns:
        Total horseshoe loss as a scalar tensor
    """
    device = next(model.parameters()).device
    horseshoe_loss = torch.tensor(0.0, device=device)

    for param in model.parameters():
        # Log-sum form for numerical stability
        # log(1 + θ²/τ²) = log(τ² + θ²) - log(τ²)
        horseshoe_loss = horseshoe_loss + torch.sum(
            torch.log(tau**2 + param**2 + epsilon) - torch.log(torch.tensor(tau**2 + epsilon))
        )

    return horseshoe_loss


def ard_loss(model: nn.Module):
    """
    Compute the ARD (Automatic Relevance Determination) loss from model modules.

    Iterates through all modules in the model and aggregates ARD loss
    contributions from modules that implement the `ard_loss` method.

    Args:
        model: Neural network module that may contain sub-modules
               with ARD regularization

    Returns:
        Total ARD loss as a scalar (0.0 if no modules implement ARD)
    """
    total = 0.0
    for m in model.modules():
        if hasattr(m, "ard_loss") and callable(m.ard_loss):
            total = total + m.ard_loss()
    return total
