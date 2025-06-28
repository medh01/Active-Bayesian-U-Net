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
display(styled)
