import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from dataLoading                import get_loaders_active
from bayesian_unet              import BayesianUNet
from acquisition_functions      import random_score, entropy, BALD, committee_kl_divergence, committee_js_divergence
from active_learning_utils      import reset_data, create_active_learning_pools, move_images_with_dict, score_unlabeled_pool
from train_eval                 import train_one_epoch, evaluate_loader

ACQ_FUNCS = {
    "random":        random_score,
    "entropy":       entropy,
    "bald":          BALD,
    "kl-divergence": committee_kl_divergence,
    "js-divergence": committee_js_divergence
}

def active_learning_loop(
        BASE_DIR: str,
        LABEL_SPLIT_RATIO: float = 0.1,
        TEST_SPLIT_RATIO: float = 0.2,
        sample_size: int = 10,
        acquisition_type: str = "random",
        mc_runs: int = 5,
        num_epochs: int = 5,
        batch_size: int = 4,
        lr: float = 1e-3,
        loop_iterations: int = None,   # None → until pool empties
        seed: int = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> pd.DataFrame:
    # 1) Reset pools & seed
    reset_data(BASE_DIR)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    # 2) Initial split
    dirs = create_active_learning_pools(
        BASE_DIR,
        LABEL_SPLIT_RATIO,
        TEST_SPLIT_RATIO,
        shuffle=True
    )
    # total training images
    total_train = (
        len(os.listdir(dirs["labeled_img"])) +
        len(os.listdir(dirs["unlabeled_img"]))
    )

    scorer = ACQ_FUNCS[acquisition_type.lower()]
    iteration = 0

    # prepare a list to collect metrics
    records = []

    while True:
        # stop on iteration limit
        if loop_iterations is not None and iteration >= loop_iterations:
            break

        # count unlabeled
        n_unl = len(os.listdir(dirs["unlabeled_img"]))
        if n_unl == 0:
            break

        iteration += 1
        print(f"\n-- Iter {iteration} | unlabeled remaining: {n_unl} --")

        # a) build loaders
        L, U, T = get_loaders_active(
            dirs["labeled_img"], dirs["labeled_mask"],
            dirs["unlabeled_img"],
            dirs["test_img"], dirs["test_mask"],
            batch_size,
            num_workers=4,
            pin_memory=True
        )

        # b) train
        model     = BayesianUNet(1,4,[64,128,256,512],0.1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_f    = nn.CrossEntropyLoss()
        for ep in range(num_epochs):
            train_one_epoch(L, model, optimizer, loss_f, device=device)

        # c) evaluate
        test_acc, test_dice = evaluate_loader(T, model, device=device, num_classes=4)
        print(f"  Test dice = {test_dice:.4f}")

        # d) record metrics
        curr_labeled = len(os.listdir(dirs["labeled_img"]))
        frac = curr_labeled / total_train
        records.append({
            "method":       acquisition_type,
            "fraction":     frac,
            "dice":         test_dice
        })

        # e) acquire & move
        score_dict = score_unlabeled_pool(U, model, scorer, T=mc_runs, num_classes=4, device=device)
        move_cnt    = min(sample_size, n_unl)
        move_images_with_dict(
            BASE_DIR, "Labeled_pool", "Unlabeled_pool",
            score_dict, num_to_move=move_cnt
        )

    # return a DataFrame
    return pd.DataFrame(records)
