"""
Loss functions and evaluation metrics for binary segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def dice_loss(logits: torch.Tensor,
              targets: torch.Tensor,
              smooth: float = 1.0) -> torch.Tensor:
    """
    Soft Dice loss.

    Parameters
    ----------
    logits  : (B, 1, H, W) raw model output (before sigmoid)
    targets : (B, 1, H, W) binary ground-truth mask
    """
    probs = torch.sigmoid(logits)
    probs_flat   = probs.view(probs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (probs_flat * targets_flat).sum(dim=1)
    denom        = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


def focal_loss(logits: torch.Tensor,
               targets: torch.Tensor,
               alpha: float = 0.8,
               gamma: float = 2.0) -> torch.Tensor:
    """
    Focal loss — down-weights easy negatives, useful for class-imbalanced
    datasets like brain tumor where foreground pixels are rare.
    """
    bce  = F.binary_cross_entropy_with_logits(logits, targets,
                                               reduction="none")
    prob  = torch.sigmoid(logits)
    pt    = targets * prob + (1 - targets) * (1 - prob)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()


class CombinedLoss(nn.Module):
    """
    Weighted sum of Focal loss + Dice loss.

    This combination works well for medical image segmentation:
    - Focal handles extreme class imbalance
    - Dice directly optimises the evaluation metric
    """

    def __init__(self, focal_weight: float = 0.5,
                 dice_weight: float = 0.5,
                 focal_alpha: float = 0.8,
                 focal_gamma: float = 2.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight  = dice_weight
        self.focal_alpha  = focal_alpha
        self.focal_gamma  = focal_gamma

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        fl = focal_loss(logits, targets,
                        alpha=self.focal_alpha, gamma=self.focal_gamma)
        dl = dice_loss(logits, targets)
        return self.focal_weight * fl + self.dice_weight * dl


# ---------------------------------------------------------------------------
# Metrics  (operate on binary predictions, not logits)
# ---------------------------------------------------------------------------

def _to_binary(logits: torch.Tensor,
               threshold: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) >= threshold).float()


def dice_score(logits: torch.Tensor,
               targets: torch.Tensor,
               threshold: float = 0.5,
               smooth: float = 1.0) -> float:
    preds  = _to_binary(logits, threshold)
    pf     = preds.view(preds.size(0), -1)
    tf     = targets.view(targets.size(0), -1)
    inter  = (pf * tf).sum(dim=1)
    denom  = pf.sum(dim=1) + tf.sum(dim=1)
    score  = ((2.0 * inter + smooth) / (denom + smooth)).mean()
    return score.item()


def iou_score(logits: torch.Tensor,
              targets: torch.Tensor,
              threshold: float = 0.5,
              smooth: float = 1.0) -> float:
    preds  = _to_binary(logits, threshold)
    pf     = preds.view(preds.size(0), -1).bool()
    tf     = targets.view(targets.size(0), -1).bool()
    inter  = (pf & tf).float().sum(dim=1)
    union  = (pf | tf).float().sum(dim=1)
    score  = ((inter + smooth) / (union + smooth)).mean()
    return score.item()


def precision_recall(logits: torch.Tensor,
                     targets: torch.Tensor,
                     threshold: float = 0.5):
    preds = _to_binary(logits, threshold)
    pf    = preds.view(-1)
    tf    = targets.view(-1)

    tp = (pf * tf).sum()
    fp = (pf * (1 - tf)).sum()
    fn = ((1 - pf) * tf).sum()

    prec = (tp / (tp + fp + 1e-8)).item()
    rec  = (tp / (tp + fn + 1e-8)).item()
    return prec, rec


class MetricTracker:
    """Accumulates per-batch metrics and returns epoch averages."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._sums   = {}
        self._counts = {}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self._sums[k]   = self._sums.get(k, 0.0) + float(v)
            self._counts[k] = self._counts.get(k, 0)  + 1

    def averages(self) -> dict:
        return {k: self._sums[k] / self._counts[k]
                for k in self._sums}

    def __str__(self):
        avgs = self.averages()
        return "  ".join(f"{k}: {v:.4f}" for k, v in avgs.items())
