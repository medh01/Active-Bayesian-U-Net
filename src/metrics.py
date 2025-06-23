import torch
import torch.nn.functional as F

def pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

    correct = (preds == targets).sum()
    total   = preds.numel()
    return correct.float() / total

def macro_dice(preds: torch.Tensor,
               targets: torch.Tensor,
               num_classes: int,
               eps: float = 1e-6) -> torch.Tensor:

    # One-hot encode to (N, C, H, W)
    p_oh = F.one_hot(preds,   num_classes).permute(0,3,1,2).float()
    t_oh = F.one_hot(targets, num_classes).permute(0,3,1,2).float()

    # Sum over batch+spatial dims → (C,)
    dims = (0, 2, 3)
    inter = (p_oh * t_oh).sum(dims)
    union = p_oh.sum(dims) + t_oh.sum(dims)

    dice_per_class = (2 * inter + eps) / (union + eps)
    return dice_per_class.mean()
