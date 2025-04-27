import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── 0) Load and combine ─────────────────────────────────────────────
df  = pd.read_csv("ai-economics-2025/data.csv")
df2 = pd.read_csv("ai-economics-2025/data2.csv")

# add / harmonise columns for the tourism sheet
df2['sector']  = 'tourism'
df2['unit']    = 'mio' + df2['unit']
df2['revenue-2024'] *= 1_000     # scale to match main sheet
df2['revenue-2022'] *= 1_000

df = pd.concat([df, df2], ignore_index=True)

EURtoUSD2022 = 1.135
EURtoUSD2024 = 1
GBPtoUSD2022 = 1.17
GBPtoUSD2024 = 1.18


def convert(row, year):
    unit = row["unit"]
    if unit == "mioGBP":
        rate = GBPtoUSD2024 if year == 2024 else GBPtoUSD2022
    elif unit == "mioEUR":
        rate = EURtoUSD2024 if year == 2024 else EURtoUSD2022
    else:  # already USD
        rate = 1
    return row[f"revenue-{year}"] * rate

df["revenue-2024"] = df.apply(convert, axis=1, year=2024).round(2)
df["revenue-2022"] = df.apply(convert, axis=1, year=2022).round(2)
df["unit"] = "mioUSD"  

df.to_markdown("./a")


missing = df['sector'].isna().sum()
assert missing == 0, f"{missing} rows still lack a sector label!"

# ── 1) Productivity levels & changes ───────────────────────────────
df['lp-2022']      = df['revenue-2022'] / df['employment-2022']
df['lp-2024']      = df['revenue-2024'] / df['employment-2024']
df['lp-change']    = df['lp-2024'] - df['lp-2022']
df['lp-change-pct']= df['lp-change'] / df['lp-2022'] * 100

# ── 2) Employment-weighted averages by (sector × adoption) ─────────
cell = (
    df.groupby(['sector','adopted-ai'])
      .agg({
          'revenue-2022':'sum',
          'employment-2022':'sum',
          'revenue-2024':'sum',
          'employment-2024':'sum'
      })
)

cell['lp-2022-w'] = cell['revenue-2022'] / cell['employment-2022']
cell['lp-2024-w'] = cell['revenue-2024'] / cell['employment-2024']
cell['lp-change-w']     = cell['lp-2024-w'] - cell['lp-2022-w']
cell['lp-change-pct-w'] = cell['lp-change-w'] / cell['lp-2022-w'] * 100

print("\nEmployment-weighted labour-productivity table")
print(cell[['lp-2022-w','lp-2024-w','lp-change-w','lp-change-pct-w']])

# ── 3) Sector-specific DiD  ────────────────────────────────────────
did_rows = []
for sec in cell.index.get_level_values(0).unique():
    try:
        treat  = cell.loc[(sec,1),'lp-change-w']
        control= cell.loc[(sec,0),'lp-change-w']
        did_rows.append({'sector':sec,'DiD_lp_change_w':treat-control})
    except KeyError:
        # one of the adoption cells missing → skip
        pass

did_df = pd.DataFrame(did_rows)
print("\nWeighted Difference-in-Differences by sector")
print(did_df.sort_values('DiD_lp_change_w', ascending=False))

# ── 4) Plot one line per (sector×adoption)  ───────────────────────-
plt.figure(figsize=(9,6))
for (sec,adopted), row in cell.iterrows():
    style = 'solid' if adopted==1 else 'dashed'
    label = f"{sec} – AI" if adopted==1 else f"{sec} – No AI"
    plt.plot([2022,2024],
             [row['lp-2022-w'], row['lp-2024-w']],
             marker='o', linestyle=style, label=label)

plt.title("Employment-Weighted Labour Productivity by Sector")
plt.xlabel("Year");      plt.ylabel("Revenue per Employee")
plt.grid(True);          plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()


# ── 3) Sector-specific DiD  + 95 % CI via bootstrap ───────────────
from tqdm.auto import tqdm                            # progress bar
N_BOOT = 10_000                                       # ≥2000 for quick test

def sector_did(df_sector):
    """Return employment-weighted DiD for one sector."""
    g = (
        df_sector.groupby('adopted-ai')
                 .agg({'revenue-2022':'sum','employment-2022':'sum',
                       'revenue-2024':'sum','employment-2024':'sum'})
    )
    if {0,1}.issubset(g.index):
        lp_22 = g['revenue-2022'] / g['employment-2022']
        lp_24 = g['revenue-2024'] / g['employment-2024']
        lp_ch = lp_24 - lp_22
        return (lp_ch.loc[1] - lp_ch.loc[0])          # treated − control
    return np.nan

did_rows = []
ci_rows  = []

for sec, df_sec in df.groupby('sector'):
    # point estimate
    point = sector_did(df_sec)
    if np.isnan(point):                               # sector lacks a group
        continue
    did_rows.append({'sector':sec, 'DiD_lp_change_w':point})

    # bootstrap distribution
    boot_vals = []
    firms = df_sec['company'].unique()               # resample at firm level
    for _ in range(N_BOOT):
        boot_idx = np.random.choice(firms, size=len(firms), replace=True)
        boot_df  = df_sec[df_sec['company'].isin(boot_idx)].copy()
        boot_vals.append(sector_did(boot_df))

    lo, hi = np.nanpercentile(boot_vals, [2.5, 97.5])
    ci_rows.append({'sector':sec,
                    'DiD_lp_change_w':point,
                    'CI_low':lo,
                    'CI_high':hi})

# tidy data-frames
did_df = pd.DataFrame(did_rows)
did_ci = pd.DataFrame(ci_rows).sort_values('DiD_lp_change_w', ascending=False)

print("\nWeighted Difference-in-Differences with 95 % CIs")
print(did_ci.to_string(index=False, float_format='%.6f'))
