# --------------------------------------------------------------
# Active Bayesian-UNet loop – no outer stopping criteria
# --------------------------------------------------------------
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
from train_eval             import train_one_epoch, evaluate_loader

ACQ_FUNCS = {
    "random":        random_score,
    "entropy":       entropy,
    "bald":          BALD,
    "kl-divergence": committee_kl_divergence,
    "js-divergence": committee_js_divergence,
}

def active_learning_loop(
        BASE_DIR: str,
        LABEL_SPLIT_RATIO: float = .1,
        TEST_SPLIT_RATIO: float = .2,
        sample_size: int = 10,
        acquisition_type: str = "js-divergence",
        mc_runs: int = 5,
        batch_size: int = 4,
        lr: float = 1e-3,
        seed: int | None = None,
        loop_iterations: int | None = None,   # set None to disable
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        # early-stopping inside each fine-tune
        patience: int = 5,
        min_delta: float = 1e-4,
):
    # ─────────────────── housekeeping ────────────────────────
    reset_data(BASE_DIR)
    g = torch.Generator()
    if seed is not None:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        g.manual_seed(seed)

    dirs = create_active_learning_pools(
        BASE_DIR, LABEL_SPLIT_RATIO, TEST_SPLIT_RATIO, shuffle=True
    )
    scorer   = ACQ_FUNCS[acquisition_type.lower()]
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ─────────────────── model built once ────────────────────
    model     = BayesianUNet(1, 4, [64,128,256,512], 0.1).to(device)
    loss_f    = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
            batch_size,
            generator=g,
            num_workers=4, pin_memory=True
        )

        # ───── fine-tune with early stopping (no epoch cap) ───
        best_val, wait, epoch = -float("inf"), 0, 0
        while True:
            epoch += 1
            train_one_epoch(L, model, optimizer, loss_f, device=device)

            model.eval()
            with torch.no_grad():
                _, val_dice = evaluate_loader(T, model, device=device,
                                              num_classes=4)
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
        _, test_dice = evaluate_loader(T, model, device=device, num_classes=4)
        curr_labeled = len(os.listdir(dirs["labeled_img"]))  # how many labelled now

        # record metrics
        frac = curr_labeled / total_train  # fraction of full data

        log.append({
            "round": iteration,
            "fraction": frac,
            "dice": test_dice,
        })

        print(f"    Round Dice = {test_dice:.4f}")

        # acquisition
        if not train_on_full_data:
            score_dict = score_unlabeled_pool(
                U, model, scorer, T=mc_runs, num_classes=4, device=device
            )
            move_images_with_dict(BASE_DIR, "Labeled_pool", "Unlabeled_pool",
                                  score_dict, num_to_move=min(sample_size, n_unl))

        # checkpoint for this round
        torch.save({"round": iteration,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "dice": test_dice},
                   os.path.join(ckpt_dir, f"active_learning_iteration_{iteration:03d}.pt"))

    return pd.DataFrame(log)
