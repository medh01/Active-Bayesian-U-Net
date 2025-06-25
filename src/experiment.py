import pandas as pd
import matplotlib.pyplot as plt
from active_learning_loop import active_learning_loop

def collect_active_learning_results(
    BASE_DIR,
    seeds,
    acquisition_funcs,
    label_split_ratio=0.1,
    test_split_ratio=0.2,
    sample_size=10,
    mc_runs=5,
    batch_size=4,
    lr=1e-3,
    loop_iterations=None,
    device="cuda"
):
    all_runs = []
    for seed in seeds:
        for acq in acquisition_funcs:
            print(f"Running acquisition={acq}, seed={seed}")
            df = active_learning_loop(
                BASE_DIR          = BASE_DIR,
                LABEL_SPLIT_RATIO = label_split_ratio,
                TEST_SPLIT_RATIO  = test_split_ratio,
                sample_size       = sample_size,
                acquisition_type  = acq,
                mc_runs           = mc_runs,
                batch_size        = batch_size,
                lr                = lr,
                loop_iterations   = loop_iterations,
                seed              = seed,
                device            = device
            )

            df["seed"]   = seed
            df["method"] = acq
            all_runs.append(df)

    overall_df = pd.concat(all_runs, ignore_index=True)
    return overall_df


def plot_active_learning_results(
    df,
    ylim=(0, 1),
    figsize=(12, 8),
    markers=None
):
    if markers is None:
        markers = {
            'random':        'o',
            'entropy':       's',
            'bald':          '^',
            'kl-divergence': 'D',
            'js-divergence': 'X',
        }

    # compute stats
    stats = (
        df
        .groupby(['method', 'fraction'])['dice']
        .agg(mean='mean', std='std')
        .reset_index()
    )

    # plot
    plt.figure(figsize=figsize)
    for method, grp in stats.groupby('method'):
        plt.errorbar(
            grp['fraction'],
            grp['mean'],
            yerr=grp['std'],
            marker=markers.get(method, 'o'),
            linewidth=2.5,
            markersize=8,
            capsize=4,
            alpha=0.8,
            label=method
        )

    plt.ylim(*ylim)
    plt.title('Test Dice vs Fraction Labelled\n(mean ± std over seeds)', fontsize=18)
    plt.xlabel('Fraction of Dataset Labelled', fontsize=14)
    plt.ylabel('Test Dice Score', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(which='major', linestyle='--', alpha=0.3)
    plt.legend(title='Acquisition', fontsize=12, title_fontsize=13,
               loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

