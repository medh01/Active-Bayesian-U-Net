import torch
import torch.nn.functional as F

def pixel_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Calculates the pixel accuracy between predicted and target masks.

    Pixel accuracy is the ratio of correctly classified pixels to the total number of pixels.

    Args:
        preds (torch.Tensor): The predicted mask tensor of shape (N, H, W).
        targets (torch.Tensor): The ground truth mask tensor of shape (N, H, W).

    Returns:
        torch.Tensor: The pixel accuracy as a scalar float tensor of shape ().
    """

    # preds and targets are of shape (N, H, W)
    correct = (preds == targets).sum() # Shape: ()
    total   = preds.numel()
    return correct.float() / total # Shape: ()

def macro_dice(preds: torch.Tensor,
               targets: torch.Tensor,
               num_classes: int,
               eps: float = 1e-6) -> torch.Tensor:
    """Calculates the macro-averaged Dice coefficient.

    The Dice coefficient is a common metric for evaluating segmentation performance.
    Macro-averaging means the Dice score is calculated for each class independently,
    and then the average of these per-class Dice scores is returned.

    Args:
        preds (torch.Tensor): The predicted mask tensor.
        targets (torch.Tensor): The ground truth mask tensor.
        num_classes (int): The total number of classes.
        eps (float, optional): A small epsilon value to prevent division by zero.
                               Defaults to 1e-6.

    Returns:
        torch.Tensor: The macro-averaged Dice coefficient as a float tensor.
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
