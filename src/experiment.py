import pandas as pd
from active_learning_loop import active_learning_loop

def collect_active_learning_results(
    BASE_DIR,
    seeds,
    acquisition_funcs,
    LABEL_SPLIT_RATIO=0.1,
    TEST_SPLIT_RATIO=0.2,
    augment=False,
    sample_size=10,
    mc_runs=5,
    batch_size=4,
    lr=1e-3,
    loop_iterations=None,
    device="cuda"
):
    """
    Collects active learning results by running the active learning loop for different seeds and acquisition functions.

    Args:
        BASE_DIR (str): The base directory for the data.
        seeds (list): A list of random seeds to use for each run.
        acquisition_funcs (list): A list of acquisition functions to test.
        LABEL_SPLIT_RATIO (float, optional): The ratio of labeled data to split from the original data. Defaults to 0.1.
        TEST_SPLIT_RATIO (float, optional): The ratio of test data to split from the original data. Defaults to 0.2.
        augment (bool, optional): Whether to use data augmentation. Defaults to False.
        sample_size (int, optional): The number of samples to acquire in each iteration. Defaults to 10.
        mc_runs (int, optional): The number of Monte Carlo runs for evaluation. Defaults to 5.
        batch_size (int, optional): The batch size for training and evaluation. Defaults to 4.
        lr (float, optional): The learning rate for the optimizer. Defaults to 1e-3.
        loop_iterations (int | None, optional): The number of active learning iterations. Defaults to None.
        device (str, optional): The device to use for training and evaluation. Defaults to "cuda".

    Returns:
        pd.DataFrame: A pandas DataFrame containing the aggregated results from all runs.
    """
    all_runs = []
    for seed in seeds:
        for acq in acquisition_funcs:
            print(f"Running acquisition={acq}")
            df = active_learning_loop(
                BASE_DIR=BASE_DIR,
                LABEL_SPLIT_RATIO=LABEL_SPLIT_RATIO,
                TEST_SPLIT_RATIO=TEST_SPLIT_RATIO,
                augment=augment,
                sample_size=sample_size,
                acquisition_type=acq,
                mc_runs=mc_runs,
                batch_size=batch_size,
                lr=lr,
                loop_iterations=loop_iterations,
                seed=seed,
                device=device
            )
            df["seed"]   = seed
            df["method"] = acq
            all_runs.append(df)

    overall_df = pd.concat(all_runs, ignore_index=True)
    return overall_df

