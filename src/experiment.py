import pandas as pd
import matplotlib.pyplot as plt

from active_learning_loop import active_learning_loop

# 1. Define your seeds and parameters
SEEDS = [0, 1, 2, 3, 4]   # 5 independent runs
ACQ_FUNCS = ["random","entropy","bald","kl-divergence","js-divergence"]

all_runs = []

for seed in SEEDS:
    for acq in ACQ_FUNCS:
        print(f"Running acquisition={acq}, seed={seed}")
        df = active_learning_loop(
            BASE_DIR          = "/path/to/data",
            LABEL_SPLIT_RATIO = 0.1,
            TEST_SPLIT_RATIO  = 0.2,
            sample_size       = 10,
            acquisition_type  = acq,
            mc_runs           = 5,
            num_epochs        = 5,
            batch_size        = 4,
            lr                = 1e-3,
            loop_iterations   = None,    # until pool empty
            seed              = seed,
            device            = "cuda"
        )
        df["seed"]   = seed
        df["method"] = acq
        all_runs.append(df)

# 2. Concatenate
big_df = pd.concat(all_runs, ignore_index=True)

# 3. Compute mean & std
stats = (
    big_df
      .groupby(["method","fraction"])["dice"]
      .agg(mean="mean", std="std")
      .reset_index()
)

# 4. Plot with error-bars
plt.figure(figsize=(10,6))
for method, grp in stats.groupby("method"):
    plt.errorbar(
        grp["fraction"],
        grp["mean"],
        yerr=grp["std"],
        marker="o",
        capsize=3,
        label=method
    )

plt.title("Test Dice vs Fraction Labelled\n(mean ± std over seeds)")
plt.xlabel("Fraction of Dataset Labelled")
plt.ylabel("Test Dice Score")
plt.ylim(0.5, 0.85)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
