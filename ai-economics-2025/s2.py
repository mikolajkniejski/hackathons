import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 0. (Re)load & harmonise data
# -----------------------------
df  = pd.read_csv("ai-economics-2025/data.csv")
df2 = pd.read_csv("ai-economics-2025/data2.csv")

# harmonise tourism sheet
df2['sector']  = 'tourism'
df2['unit']    = 'mio' + df2['unit']
df2['revenue-2024'] *= 1_000
df2['revenue-2022'] *= 1_000
df = pd.concat([df, df2], ignore_index=True)

# simple currency conversion to USD
EURtoUSD2022, EURtoUSD2024 = 1.135, 1.0
GBPtoUSD2022, GBPtoUSD2024 = 1.17, 1.18
def convert(row, year):
    unit = row["unit"]
    if unit == "mioGBP":
        rate = GBPtoUSD2024 if year == 2024 else GBPtoUSD2022
    elif unit == "mioEUR":
        rate = EURtoUSD2024 if year == 2024 else EURtoUSD2022
    else:
        rate = 1
    return row[f"revenue-{year}"] * rate

df["revenue-2024"] = df.apply(convert, axis=1, year=2024).round(2)
df["revenue-2022"] = df.apply(convert, axis=1, year=2022).round(2)
df["unit"] = "mioUSD"

# -----------------------------
# 1. Productivity metrics
# -----------------------------
df['lp-2022']       = df['revenue-2022'] / df['employment-2022']
df['lp-2024']       = df['revenue-2024'] / df['employment-2024']
df['lp-change']     = df['lp-2024'] - df['lp-2022']
df['lp-change-pct'] = df['lp-change'] / df['lp-2022'] * 100

# weighted aggregates
cell = (
    df.groupby(['sector', 'adopted-ai'])
      .agg({'revenue-2022':'sum', 'employment-2022':'sum',
            'revenue-2024':'sum', 'employment-2024':'sum'})
)

cell['lp-2022-w'] = cell['revenue-2022'] / cell['employment-2022']
cell['lp-2024-w'] = cell['revenue-2024'] / cell['employment-2024']
cell['lp-change-w']     = cell['lp-2024-w'] - cell['lp-2022-w']
cell['lp-change-pct-w'] = cell['lp-change-w'] / cell['lp-2022-w'] * 100

# -----------------------------
# 2. Extra Plots
# -----------------------------

# (a) Bar plot: % change in LP by sector & adoption
pivot_pct = cell['lp-change-pct-w'].unstack('adopted-ai')
labels = pivot_pct.index.tolist()
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8,5))
ax.bar(x - width/2, pivot_pct[0], width, label='No AI')
ax.bar(x + width/2, pivot_pct[1], width, label='AI')

ax.set_ylabel('% change in revenue/employee')
ax.set_title('Employment-weighted LP growth (2022→24)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('/mnt/data/plot_bar_lp_growth.png')
plt.show()

# (b) Error-bar plot: DiD estimates with 95 % CI
def sector_did(df_sector):
    g = (df_sector.groupby('adopted-ai')
                  .agg({'revenue-2022':'sum','employment-2022':'sum',
                        'revenue-2024':'sum','employment-2024':'sum'}))
    if {0,1}.issubset(g.index):
        lp22 = g['revenue-2022']/g['employment-2022']
        lp24 = g['revenue-2024']/g['employment-2024']
        return (lp24 - lp22).loc[1] - (lp24 - lp22).loc[0]
    return np.nan

N_BOOT = 4000  # a bit lighter for speed
did_stats = []
for sector, df_sec in df.groupby('sector'):
    point = sector_did(df_sec)
    if np.isnan(point): 
        continue
    boot = []
    firms = df_sec['company'].unique()
    for _ in range(N_BOOT):
        sample_ids = np.random.choice(firms, size=len(firms), replace=True)
        boot_df = df_sec[df_sec['company'].isin(sample_ids)].copy()
        boot.append(sector_did(boot_df))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    did_stats.append({'sector':sector, 'point':point, 'lo':lo, 'hi':hi})

did_df = pd.DataFrame(did_stats).sort_values('point', ascending=False)

# error-bar chart
fig, ax = plt.subplots(figsize=(6,4))
ax.errorbar(did_df['sector'], did_df['point'], 
            yerr=[did_df['point']-did_df['lo'], did_df['hi']-did_df['point']],
            fmt='o', capsize=5)
ax.axhline(0, color='grey', linestyle='--')
ax.set_ylabel('Δ labour productivity (USD/employee)')
ax.set_title('Difference-in-Differences (AI vs. No-AI, 2022→24)')
plt.tight_layout()
plt.savefig('/mnt/data/plot_did_ci.png')
plt.show()

# (c) Scatter plot: firm-level LP change vs adoption
fig, ax = plt.subplots(figsize=(8,5))
colors = {0:'tab:blue', 1:'tab:orange'}
for adopted, grp in df.groupby('adopted-ai'):
    ax.scatter(grp['lp-change-pct'], grp['sector'], 
               alpha=0.6, label=('AI' if adopted else 'No AI'))
ax.set_xlabel('% change in revenue/employee (firm-level)')
ax.set_title('Distribution of firm-level LP changes by adoption')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('/mnt/data/plot_scatter_firm_lp.png')
plt.show()

print("Plots saved to /mnt/data/:\n  - plot_bar_lp_growth.png\n  - plot_did_ci.png\n  - plot_scatter_firm_lp.png")
