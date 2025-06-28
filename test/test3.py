import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

base_dir = '../experiments'
file_paths = glob.glob(os.path.join(base_dir, '*', '*_log.csv'))

# Read + concat
dfs = []
for fp in file_paths:
    df = pd.read_csv(fp)
    df['seed'] = os.path.basename(os.path.dirname(fp))
    dfs.append(df)
df_all = pd.concat(dfs, ignore_index=True)

# normalize column name
if 'fraction of the dataset' in df_all.columns:
    df_all = df_all.rename(columns={'fraction of the dataset': 'fraction'})

# mean dice per method/fraction
df_mean = df_all.groupby(['method', 'fraction'], as_index=False)['dice'].mean()

# define line‐styles for each method
style_map = {
    'random':               '-',   # solid
    'entropy':              ':',   # dotted
    'bald':                 '--',  # dashed
    'kl-divergence':        '-.',  # dash-dot
    'js-divergence':        (0, (5, 1)),  # custom dash pattern
}

plt.figure(figsize=(10,8))
for method in df_mean['method'].unique():
    sub = df_mean[df_mean['method'] == method].sort_values('fraction')
    ls = style_map.get(method.lower(), '-')  # fallback to solid
    plt.plot(sub['fraction'], sub['dice'],
             linestyle=ls,
             linewidth=2,
             label=method)

plt.xlabel('Fraction of dataset')
plt.ylabel('Mean Dice score')
plt.title('Dice Score vs Dataset Fraction by Acquisition Function')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd

# assume you’ve already read & concatenated all logs into df_all
# and renamed the fraction column to ‘fraction’, then computed:
# df_mean = df_all.groupby(['method','fraction'], as_index=False)['dice'].mean()

# 1) Define the quartiles you want to show
quartiles = [0.25, 0.50, 0.75, 1.00]

# 2) Compute the raw quantile values
quantile_vals = df_mean['fraction'].quantile(quartiles).values

# 3) Snap each quantile to the nearest actual fraction in your data
available = sorted(df_mean['fraction'].unique())
selected = sorted({
    min(available, key=lambda x: abs(x - q))
    for q in quantile_vals
})

# 4) Subset & pivot into a table
df_sub = df_mean[df_mean['fraction'].isin(selected)]
table = df_sub.pivot(index='fraction', columns='method', values='dice')

# 5) Format the index as percentages
#    (if your fractions are 0–1; divide by 1, then *100)
table.index = [f"{int(frac*100)} %" for frac in table.index]

# 6) (Optional) Reorder the columns to your preferred method order
table = table[['random','entropy','bald','js-divergence','kl-divergence']]

# 7) Display and highlight the best score in each row
styled = (
    table
      .round(3)                            # three sig-figs
      .style
      .format("{:.3f}")                    # ensure trailing zeros
      .highlight_max(axis=1, props="font-weight: bold;")
)
from IPython.display import display
print(table)
