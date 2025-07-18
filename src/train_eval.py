import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from metrics import pixel_accuracy, macro_dice


def train_one_epoch(loader,
                    model,
                    optimizer,
                    loss_fn,
                    device="cuda",
                    scheduler=None,  # any torch.optim.lr_scheduler.*
                    scaler=None,  # GradScaler(); will create one if None
                    num_classes=5):
    """Trains the model for one epoch.

    This function iterates over the training data, performs forward and backward passes,
    and updates the model's weights. It also calculates and displays training metrics.

    Args:
        loader (DataLoader): The data loader for the training set.
        model (nn.Module): The model to be trained.
        optimizer (Optimizer): The optimizer for updating the model's weights.
        loss_fn (callable): The loss function.
        device (str, optional): The device to train on. Defaults to "cuda".
        scheduler (lr_scheduler, optional): A learning rate scheduler. Defaults to None.
        scaler (GradScaler, optional): A gradient scaler for mixed-precision training. Defaults to None.
        num_classes (int, optional): The number of classes for metric calculation. Defaults to 5.

    Returns:
        tuple: A tuple containing the mean loss, mean accuracy, and mean Dice coefficient for the epoch.
    """

    model.train()  # Puts modules such as Dropout and BatchNorm into training mode
    device = torch.device(device)
    scaler = scaler or GradScaler()  # performance
    tot_loss, tot_acc, tot_dice = 0.0, 0.0, 0.0
    n_batches = 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for imgs, masks, *_ in pbar:
        imgs = imgs.float().to(device, non_blocking=True)
        masks = masks.long().to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(imgs)  # (N, C, H, W)
            loss = loss_fn(logits, masks)  # scalar

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)  # label map (N, H, W)

            batch_acc = pixel_accuracy(preds, masks)
            batch_dice = macro_dice(preds, masks, num_classes)

        n_batches += 1
        tot_loss += loss.item()
        tot_acc += batch_acc.item()
        tot_dice += batch_dice.item()

        pbar.set_postfix({
            "loss": tot_loss / n_batches,
            "acc%": (tot_acc / n_batches) * 100,
            "dice": tot_dice / n_batches,
            "lr": optimizer.param_groups[0]["lr"]
        })

    pbar.close()

    if scheduler and isinstance(scheduler,
                                torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(tot_loss / n_batches)

    mean_loss = tot_loss / n_batches
    mean_acc = tot_acc / n_batches * 100.0
    mean_dice = tot_dice / n_batches
    return mean_loss, mean_acc, mean_dice


@torch.no_grad()
def evaluate_loader(loader, model, device="cuda", num_classes=4):
    """Evaluates the model on a given data loader.

    This function iterates over the data, makes predictions, and calculates evaluation metrics.
    It operates in `torch.no_grad()` mode to save memory and computation.

    Args:
        loader (DataLoader): The data loader for the evaluation set.
        model (nn.Module): The model to be evaluated.
        device (str, optional): The device to evaluate on. Defaults to "cuda".
        num_classes (int, optional): The number of classes for metric calculation. Defaults to 4.

    Returns:
        tuple: A tuple containing the mean accuracy and mean Dice coefficient.
    """
    model.eval()  # inference mode
    device = torch.device(device)

    tot_acc, tot_dice, n_batches = 0.0, 0.0, 0

    pbar = tqdm(loader, desc="Eval", leave=False)
    for imgs, masks, *_ in pbar:
        imgs = imgs.float().to(device, non_blocking=True)
        masks = masks.long().to(device, non_blocking=True)

        logits = model(imgs)  # (N, C, H, W)
        preds = logits.argmax(1)  # (N, H, W)

        batch_acc = pixel_accuracy(preds, masks)
        batch_dice = macro_dice(preds, masks, num_classes)

        # accumulate
        tot_acc += batch_acc.item()
        tot_dice += batch_dice.item()
        n_batches += 1

        pbar.set_postfix({"acc%": tot_acc / n_batches,
                          "dice": tot_dice / n_batches})
    pbar.close()

    if n_batches == 0:
        raise ValueError("Loader is empty – nothing to evaluate.")

    return tot_acc / n_batches, tot_dice / n_batches
