import torch
import torch.nn.functional as F
from torch import nn

class TverskyLossNoBG(nn.Module):
    """A Tversky loss function that excludes the background class from the calculation.

    The Tversky index is a generalization of the Dice coefficient and the Jaccard index.
    This implementation is designed for multi-class segmentation and allows for weighting
    false positives and false negatives through the `alpha` and `beta` parameters.
    The background class is ignored in the loss calculation.

    Args:
        alpha (float): The weight for false positives.
        beta (float): The weight for false negatives.
        smooth (float, optional): A smoothing factor to avoid division by zero. Defaults to 1e-6.
        bg_idx (int, optional): The index of the background class to be ignored. Defaults to 4.
    """
    def __init__(self, alpha, beta, smooth=1e-6, bg_idx=4):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.smooth = smooth
        self.bg_idx = bg_idx

    def forward(self, logits, target):
        B,C,H,W = logits.shape
        probs   = F.softmax(logits, dim=1)
        oh      = F.one_hot(target.clamp(0,C-1), C).permute(0,3,1,2).float()
        dims    = (0,2,3)
        TP = (probs * oh).sum(dims)
        FP = (probs * (1 - oh)).sum(dims)
        FN = ((1 - probs) * oh).sum(dims)
        # exclude background
        mask = torch.ones(C, dtype=torch.bool, device=logits.device)
        mask[self.bg_idx] = False
        TP,FP,FN = TP[mask], FP[mask], FN[mask]
        TI = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        return (1 - TI).mean()

def pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Calculates the pixel accuracy for a batch of predictions.

    Args:
        preds (torch.Tensor): The predicted segmentation maps, of shape (N, H, W).
        targets (torch.Tensor): The ground truth segmentation maps, of shape (N, H, W).

    Returns:
        torch.Tensor: A scalar tensor representing the pixel accuracy.
    """
    # preds and targets are of shape (N, H, W)
    correct = (preds == targets).sum() # Shape: ()
    total   = preds.numel()
    return correct.float() / total # Shape: ()

def macro_dice(preds: torch.Tensor,
               targets: torch.Tensor,
               num_classes: int,
               eps: float = 1e-6) -> torch.Tensor:
    """Calculates the macro-averaged Dice coefficient for a batch of predictions.

    Args:
        preds (torch.Tensor): The predicted segmentation maps, of shape (N, H, W).
        targets (torch.Tensor): The ground truth segmentation maps, of shape (N, H, W).
        num_classes (int): The total number of classes.
        eps (float, optional): A small epsilon value to avoid division by zero. Defaults to 1e-6.

    Returns:
        torch.Tensor: A scalar tensor representing the macro-averaged Dice coefficient.
    """
    # preds and targets are of shape (N, H, W)
    # One-hot encode to (N, H, W, C)
    p_oh = F.one_hot(preds,   num_classes).permute(0,3,1,2).float() # Shape: (N, H, W, C)
    t_oh = F.one_hot(targets, num_classes).permute(0,3,1,2).float() # Shape: (N, H, W, C)

    # Sum over batch+spatial dims → (C,)
    dims = (0, 2, 3)
    inter = (p_oh * t_oh).sum(dims) # Shape: (C,)
    union = p_oh.sum(dims) + t_oh.sum(dims) # Shape: (C,)

    dice_per_class = (2 * inter + eps) / (union + eps) # Shape: (C,)
    return dice_per_class.mean() # Shape: ()