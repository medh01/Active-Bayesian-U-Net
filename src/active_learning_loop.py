import os, random
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import pandas as pd


from data_loading           import get_loaders_active
from bayesian_unet          import BayesianUNet
from acquisition_functions  import (random_score, entropy, BALD,
                                    committee_kl_divergence,
                                    committee_js_divergence)
from active_learning_utils  import (reset_data, create_active_learning_pools,
                                    move_images_with_dict, score_unlabeled_pool)
from metrics import TverskyLossNoBG
from train_eval import train_one_epoch, evaluate_loader

ACQ_FUNCS = {
    "random": random_score,
    "entropy": entropy,
    "bald": BALD,
    "kl-divergence": committee_kl_divergence,
    "js-divergence": committee_js_divergence,
}


def active_learning_loop(
        BASE_DIR: str,
        LABEL_SPLIT_RATIO: float = .1,
        TEST_SPLIT_RATIO: float = .2,
        augment: bool = False,
        sample_size: int = 2,
        acquisition_type: str = "js-divergence",
        mc_runs: int = 8,
        dropout = 0.3,
        batch_size: int = 16,
        lr: float = 1e-3,
        seed: int | None = None,
        loop_iterations: int | None = None,  # set None to disable
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        # early-stopping inside each fine-tune
        patience: int = 15,
        min_delta: float = 1e-4,
):
    """Performs an active learning loop for semantic segmentation.

    This function orchestrates the entire active learning process. It initializes the data pools,
    trains a model on the labeled data, selects new samples from the unlabeled pool using an
    acquisition function, moves them to the labeled pool, and repeats the process.

    Args:
        BASE_DIR (str): The base directory for the data and checkpoints.
        LABEL_SPLIT_RATIO (float, optional): The initial ratio of labeled data. Defaults to .1.
        TEST_SPLIT_RATIO (float, optional): The ratio of data to be used for the test set. Defaults to .2.
        augment (bool, optional): Whether to use data augmentation. Defaults to False.
        sample_size (int, optional): The number of samples to acquire in each iteration. Defaults to 2.
        acquisition_type (str, optional): The acquisition function to use. Defaults to "js-divergence".
        mc_runs (int, optional): The number of Monte Carlo runs for acquisition functions. Defaults to 8.
        dropout (float, optional): The dropout probability for the model. Defaults to 0.3.
        batch_size (int, optional): The batch size for training and evaluation. Defaults to 16.
        lr (float, optional): The learning rate for the optimizer. Defaults to 1e-3.
        seed (int | None, optional): The random seed for reproducibility. Defaults to None.
        loop_iterations (int | None, optional): The maximum number of active learning iterations. Defaults to None.
        device (str, optional): The device to use for training. Defaults to "cuda" if available.
        patience (int, optional): The patience for early stopping during fine-tuning. Defaults to 15.
        min_delta (float, optional): The minimum change in validation Dice to reset patience. Defaults to 1e-4.

    Returns:
        pd.DataFrame: A DataFrame containing the log of the active learning experiment.
    """
    # ─────────────────── housekeeping ────────────────────────
    reset_data(BASE_DIR)

    g = torch.Generator()
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        g.manual_seed(seed)

    dirs = create_active_learning_pools(
        BASE_DIR, LABEL_SPLIT_RATIO, TEST_SPLIT_RATIO, shuffle=True
    )
    scorer = ACQ_FUNCS[acquisition_type.lower()]
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ─────────────────── model built once ────────────────────
    model = BayesianUNet(in_channels=1, num_classes=5, dropout_prob=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    iteration = 0
    log: list[dict] = []
    total_train = (
            len(os.listdir(dirs["labeled_img"])) +  # currently labelled
            len(os.listdir(dirs["unlabeled_img"]))  # plus still un-labelled
    )
    train_on_full_data = False
    # ─────────────────── big loop ────────────────────────────
    while True:

        # optional hard iteration cap
        if loop_iterations is not None and iteration >= loop_iterations:
            break

        # stop only when pool empty
        n_unl = len(os.listdir(dirs["unlabeled_img"]))
        if n_unl == 0:
            if train_on_full_data:
                print("Finished Training on the whole dataset")
                break
            else:
                print("Un-labelled pool exhausted")
                train_on_full_data = True

        iteration += 1
        print(f"\n── Active Learning Iteration: {iteration} | Unlabelled pool size: {n_unl}")

        # loaders
        L, U, T = get_loaders_active(
            dirs["labeled_img"], dirs["labeled_mask"],
            dirs["unlabeled_img"],
            dirs["test_img"], dirs["test_mask"],
            batch_size=batch_size,
            seed=seed,
            augment=augment,
            generator=g,
            num_workers=4, pin_memory=True
        )

        # ───── fine-tune with early stopping (no epoch cap) ───
        best_val, wait, epoch = -float("inf"), 0, 0

        # Calculate weights for cross entropy loss function
        num_classes = 5
        class_counts = torch.zeros(num_classes, dtype=torch.float32, device=device)
        total_pixels = 0

        for imgs, masks, _ in L:
            flat = masks.view(-1).to(device)
            class_counts += torch.bincount(flat, minlength=num_classes).float()
            total_pixels += flat.numel()
        class_freqs = class_counts / total_pixels
        median_freq = torch.median(class_freqs)
        weights = median_freq / class_freqs

        weights = weights / weights.mean()

        # Define Loss Function
        tversky_fn = TverskyLossNoBG(0.3, 0.7, bg_idx=4).to(device)
        ce_loss = nn.CrossEntropyLoss(weight=weights)

        def combined_loss(logits, targets):
            ce = ce_loss(logits, targets)
            tv = tversky_fn(logits, targets)
            return ce + tv

        while True:
            epoch += 1
            train_one_epoch(L, model, optimizer, combined_loss, device=device)

            model.eval()
            with torch.no_grad():
                _, val_dice = evaluate_loader(T, model, device=device,
                                              num_classes=5)
            model.train()
            print(f"    Epoch {epoch:03d} | val Dice {val_dice:.4f}")

            if val_dice > best_val + min_delta:
                best_val, wait = val_dice, 0
                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir, "best_tmp.pt"))
            else:
                wait += 1
                if wait >= patience:
                    print(f"    Early-stop after {epoch} epochs")
                    break

        model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best_tmp.pt")))

        # evaluate & log
        _, test_dice = evaluate_loader(T, model, device=device, num_classes=5)

        curr_labeled = len(os.listdir(dirs["labeled_img"]))  # how many labelled now

        # record metrics
        frac = curr_labeled / total_train  # fraction of full data

        log.append({
            "round": iteration,
            "fraction": frac,
            "dice_score": test_dice,
        })

        print(f"[Active Learning iteration: {iteration}]")
        print(f"   Validation Dice = {test_dice:.4f}")

        # acquisition
        if not train_on_full_data:
            score_dict = score_unlabeled_pool(
                U, model, scorer, T=mc_runs, num_classes=5, device=device
            )
            move_images_with_dict(BASE_DIR, "Labeled_pool", "Unlabeled_pool",
                                  score_dict, num_to_move=min(sample_size, n_unl))

    return pd.DataFrame(log)