"""
src/segmentation/metrics.py
Segmentation metrics: Dice, Jaccard (IoU), pixel accuracy.
All metrics are computed per-image and averaged (not flattened globally).
"""

import torch


def dice_score(pred_logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-8) -> float:
    """Per-image Dice averaged over the batch."""
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps)).mean().item()


def jaccard_score(pred_logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-8) -> float:
    """Per-image Jaccard (IoU) averaged over the batch."""
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + eps) / (union + eps)).mean().item()


def pixel_accuracy(pred_logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Per-image pixel accuracy averaged over the batch."""
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    correct = (pred == target).float()
    return correct.mean().item()


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Differentiable Dice loss (operates on raw logits via sigmoid)."""
    pred_sig = torch.sigmoid(pred).view(-1)
    target_flat = target.view(-1)
    intersection = (pred_sig * target_flat).sum()
    return 1 - (2 * intersection + smooth) / (pred_sig.sum() + target_flat.sum() + smooth)


def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE + Dice combined loss."""
    import torch.nn as nn
    return nn.BCEWithLogitsLoss()(pred, target) + dice_loss(pred, target)
