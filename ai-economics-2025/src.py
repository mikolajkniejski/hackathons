import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("ai-economics-2025/data.csv")
df2 = pd.read_csv("ai-economics-2025/data2.csv")
df2['sector'] = 'tourism'
df2['unit'] = 'mio'+ df2['unit']
df2['revenue-2024'] =df2['revenue-2024'] * 1000
df2['revenue-2022'] =df2['revenue-2022'] * 1000 
dfx = pd.concat([df, df2], ignore_index=True)
df = dfx

# Compute labor productivity for 2022 and 2024
df['lp-2022'] = df['revenue-2022'] / df['employment-2022']
df['lp-2024'] = df['revenue-2024'] / df['employment-2024']

# Compute change in labor productivity (absolute and %)
df['lp-change'] = df['lp-2024'] - df['lp-2022']
df['lp-change-pct'] = df['lp-change'] / df['lp-2022'] * 100

# JPYperUSD2022 = 132
# JPYperUSD2022 = 150
# BX,4695,4895,12663,8016,mioUSD,1,https://stockanalysis.com/stocks/bx/financials/
# NFLX,12800,14000,39000,31000,mioUSD,0,other,https://stockanalysis.com/quote/neo/NFLX/financials/
# BA,156000,172000,66000,66000,mioUSD,0,other,https://stockanalysis.com/stocks/ba
# DIS,204000,214000,91000,83000,mioUSD,0,other,https://stockanalysis.com/quote/bmv
# PwC,328000,370000,50300,55400,mioUSD,1,service,null
# UBS,72597,110000,48000,34000,mioUSD,1,service,https://stockanalysis.com/stocks/ubs/financials/
# ALV,159000,156000,107000,96000,mioEUR,0,insurance,https://stockanalysis.com/quote/etr/ALV/financials/
# ITC,23829,37312,606000,708000,mioUSD,0,tobacco,https://stockanalysis.com/quote/nse/ITC/financials/
# TCOM,32202,36249,53.29,44.51,CNY,1,https://stockanalysis.com/stocks/tcom/
# PUK,14196,15412,12260,11040,mioUSD,1,insurance,https://stockanalysis.com/stocks/puk/

# Group by adoption status
group_means = df.groupby('adopted-ai')[['lp-2022', 'lp-2024', 'lp-change', 'lp-change-pct']].mean().reset_index()

# Print average labor productivity by group
print("Average labor productivity by group:")
print(group_means)
df.to_markdown("./a")

# Calculate Difference-in-Differences (DiD)
treated_diff = group_means.loc[group_means['adopted-ai'] == 1, 'lp-change'].values[0]
control_diff = group_means.loc[group_means['adopted-ai'] == 0, 'lp-change'].values[0]

did_estimate = treated_diff - control_diff

print("\n--- Difference-in-Differences Estimate ---")
print(f"DiD estimate (absolute change in labor productivity): {did_estimate:.4f}")

# (Optional) Plot results
plt.figure(figsize=(8,5))
for adopted in [0, 1]:
    subset = df[df['adopted-ai'] == adopted]
    avg_lp_2022 = subset['lp-2022'].mean()
    avg_lp_2024 = subset['lp-2024'].mean()
    label = "AI Adopters" if adopted == 1 else "Non-Adopters"
    plt.plot([2022, 2024], [avg_lp_2022, avg_lp_2024], marker='o', label=label)

plt.title("Average Labor Productivity Over Time")
plt.xlabel("Year")
plt.ylabel("Labor Productivity (Revenue per Employee)")
plt.legend()
plt.grid(True)
plt.show()
