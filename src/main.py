import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

# 1) Map each method to its CSV
files = {
    'random':        '../experiments/random_log.csv',
    'entropy':       '../experiments/entropy_log.csv',
    'bald':          '../experiments/bald_log.csv',
    'js-divergence': '../experiments/js-divergence_log.csv',
    'kl-divergence': '../experiments/kl-divergence_log.csv',
}

# 2) Load & tag
df_list = []
for method, path in files.items():
    tmp = pd.read_csv(path, usecols=['fraction','dice_score'])
    tmp['method'] = method
    df_list.append(tmp)

# 3) Combine & pivot
df    = pd.concat(df_list, ignore_index=True)
pivot = df.pivot(index='fraction', columns='method', values='dice_score')

# 4) Compute mean Dice and AUC for each method
mean_dice = pivot.mean()
auc_scores = {m: np.trapz(pivot[m].values, pivot.index.values)
              for m in pivot.columns}

# 5) Print the “key fractions” table (as before)…

# 6) Now build the summary table for mean & AUC
console = Console()
summary = Table(title="Acquisition Method Summary", show_lines=True)

summary.add_column("Method",        style="bold cyan")
summary.add_column("Mean Dice",     justify="right", style="green")
summary.add_column("AUC", justify="right", style="magenta")

for method in pivot.columns:
    summary.add_row(
        method,
        f"{mean_dice[method]:.3f}",
        f"{auc_scores[method]:.3f}"
    )

console.print(summary)