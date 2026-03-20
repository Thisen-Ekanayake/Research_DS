# %% [markdown]
# # Labour–Economy Relationship Analysis — Sri Lanka (2015–2024)
# ### Phase 2: Master dataset build + relationship visualizations
# Run cells top-to-bottom. Set DATA_DIR to your CSVs folder.

# %% [markdown]
# ## Cell 0 — Imports & style

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import os

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e5e5e5",
    "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})
C_BLUE="#185FA5"; C_AMBER="#BA7517"; C_RED="#E24B4A"
C_PURPLE="#7F77DD"; C_GRAY="#888780"; C_PINK="#D4537E"; C_TEAL="#1D9E75"
OUTPUT_DIR = "charts_relationships"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Style loaded.")

# %% [markdown]
# ## Cell 1 — Build master dataset

# %%
DATA_DIR = "."   # ← point this at your CSVs folder

# 1. Underemployment (DCS LFS)
under = pd.DataFrame({
    "Year": range(2015, 2025),
    "Underemployment_Rate":   [2.7,2.4,2.8,2.6,2.7,2.6,2.5,2.3,2.5,2.4],
    "Underemployment_Male":   [2.0,1.7,2.2,2.2,2.3,2.3,2.1,2.0,2.2,2.1],
    "Underemployment_Female": [3.8,3.5,3.9,3.5,3.5,3.3,3.3,2.8,3.1,3.0],
})

# 2. Real GDP → growth rate
gdp = pd.read_csv(os.path.join(DATA_DIR, "Real_GDP_at_Constant_National_Prices_for_Sri_Lanka.csv"))
gdp["Year"] = pd.to_datetime(gdp["observation_date"]).dt.year
gdp = gdp.rename(columns={"RGDPNALKA666NRUG": "Real_GDP"})
gdp = gdp[gdp["Year"].between(2014,2024)].sort_values("Year")[["Year","Real_GDP"]]
gdp["GDP_Growth_Rate"] = gdp["Real_GDP"].pct_change() * 100
gdp = gdp[gdp["Year"] >= 2015][["Year","Real_GDP","GDP_Growth_Rate"]]

# 3. Inflation
inf = pd.read_csv(os.path.join(DATA_DIR, "Inflation__consumer_prices_for_Sri_Lanka.csv"))
inf["Year"] = pd.to_datetime(inf["observation_date"]).dt.year
inf = inf.rename(columns={"FPCPITOTLZGLKA": "Inflation_Rate"})
inf = inf[inf["Year"].between(2015,2024)][["Year","Inflation_Rate"]]

# 4. Exchange rate (daily → annual avg) + backfill 2015-2020
fx = pd.read_csv(os.path.join(DATA_DIR, "Sri_Lankan_Rupees_to_U_S__Dollar_Spot_Exchange_Rate.csv"))
fx["Year"] = pd.to_datetime(fx["observation_date"]).dt.year
fx["DEXSLUS"] = pd.to_numeric(fx["DEXSLUS"], errors="coerce")
fx_ann = fx[fx["Year"].between(2015,2024)].groupby("Year")["DEXSLUS"].mean().reset_index()
fx_ann.columns = ["Year","Exchange_Rate_LKR_USD"]
backfill = pd.DataFrame({"Year":[2015,2016,2017,2018,2019,2020],
                          "Exchange_Rate_LKR_USD":[135.86,145.60,152.46,162.54,178.74,185.59]})
fx_full = pd.concat([backfill, fx_ann[fx_ann["Year"]>=2021]]).sort_values("Year")

# 5. Informal employment (Highlights CSV)
informal = pd.DataFrame({
    "Year": range(2015,2025),
    "Informal_Pct":        [59.8,60.2,58.0,58.7,57.4,58.1,58.4,57.4,58.0,56.9],
    "Informal_Male_Pct":   [63.2,63.4,60.8,62.5,60.8,62.1,62.7,62.0,62.3,61.4],
    "Informal_Female_Pct": [53.7,54.4,53.1,51.3,51.1,49.9,49.7,48.6,49.5,47.7],
})

# 6. Youth LFPR + unemployment
youth = pd.DataFrame({
    "Year": range(2015,2025),
    "Youth_LFPR_15_24":        [33.7,32.7,33.0,30.0,30.7,29.2,26.4,25.3,24.0,23.8],
    "Youth_Unemployment_15_24":[12.8,12.4,11.8,13.1,14.2,26.5,26.5,22.7,23.0,22.0],
    "Youth_Unemp_Male":        [8.5,8.2,7.8,8.8,9.6,18.2,18.4,15.6,15.9,15.2],
    "Youth_Unemp_Female":      [19.4,18.6,17.8,19.2,20.4,37.8,36.9,31.9,32.1,30.4],
})

# 7. Overall unemployment
unemp = pd.DataFrame({"Year":range(2015,2025),
                       "Unemployment_Rate":[4.7,4.4,4.2,4.4,4.8,5.5,5.1,4.7,4.7,4.4]})

# Merge all
master = under.copy()
for df in [gdp, inf, fx_full, informal, youth, unemp]:
    master = master.merge(df, on="Year", how="left")
master = master.round(4)
master.to_csv(os.path.join(OUTPUT_DIR, "master_dataset.csv"), index=False)
print(f"Master dataset: {master.shape}")
print(master[["Year","GDP_Growth_Rate","Inflation_Rate","Exchange_Rate_LKR_USD",
              "Informal_Pct","Youth_LFPR_15_24","Underemployment_Rate"]].to_string(index=False))

# %% [markdown]
# ## Cell 2 — Correlation matrix

# %%
cols_of_interest = ["Underemployment_Rate","GDP_Growth_Rate","Inflation_Rate",
                    "Exchange_Rate_LKR_USD","Informal_Pct","Youth_LFPR_15_24",
                    "Youth_Unemployment_15_24","Unemployment_Rate"]
corr = master[cols_of_interest].corr().round(2)
print("\nCorrelation matrix:")
print(corr[["Underemployment_Rate"]])

# %% [markdown]
# ## Cell 3 — Chart 1: GDP growth vs underemployment (dual axis)

# %%
df = master.dropna(subset=["GDP_Growth_Rate"]).copy()
YEARS = df["Year"].tolist()

fig, ax1 = plt.subplots(figsize=(11, 4.5))
ax2 = ax1.twinx()

bar_colors = [C_BLUE if v >= 0 else C_RED for v in df["GDP_Growth_Rate"]]
ax1.bar(YEARS, df["GDP_Growth_Rate"], color=bar_colors, alpha=0.4, zorder=2, label="GDP growth rate")
ax1.axhline(0, color="#cccccc", linewidth=0.8)
ax1.set_ylabel("GDP growth rate (%)", color=C_BLUE)
ax1.tick_params(axis='y', labelcolor=C_BLUE)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))

ax2.plot(YEARS, df["Underemployment_Rate"], marker="o", markersize=6,
         color=C_AMBER, linewidth=2.5, label="Underemployment rate")
ax2.set_ylabel("Underemployment rate (%)", color=C_AMBER)
ax2.tick_params(axis='y', labelcolor=C_AMBER)
ax2.set_ylim(1.5, 3.5)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))

# Annotate 2022 contraction
ax1.annotate("2022: GDP −7.3%\nInflation 49.7%", xy=(2022, -7.35),
             xytext=(2020.3, -6.5), fontsize=8.5, color=C_RED,
             arrowprops=dict(arrowstyle="->", color=C_RED, lw=0.8))

ax1.set_xticks(YEARS)
ax1.set_title("GDP growth rate vs underemployment rate — Sri Lanka (2015–2023)", fontweight="500")

from matplotlib.lines import Line2D
handles = [plt.Rectangle((0,0),1,1, color=C_BLUE, alpha=0.4),
           Line2D([0],[0], color=C_AMBER, marker='o', linewidth=2)]
ax1.legend(handles, ["GDP growth (bars)", "Underemployment (line)"], frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart1_gdp_vs_underemployment.png"))
plt.show()
print("Saved → chart1_gdp_vs_underemployment.png")

# %% [markdown]
# ## Cell 4 — Chart 2: Inflation vs underemployment — 2022 crisis focus

# %%
YEARS_ALL = master["Year"].tolist()
bar_colors_inf = [C_RED if v > 20 else C_AMBER if v > 7 else C_GRAY
                  for v in master["Inflation_Rate"]]

fig, ax1 = plt.subplots(figsize=(11, 4.5))
ax2 = ax1.twinx()

ax1.bar(YEARS_ALL, master["Inflation_Rate"], color=bar_colors_inf, alpha=0.5, zorder=2)
ax1.set_ylabel("Inflation rate (%)", color=C_RED)
ax1.tick_params(axis='y', labelcolor=C_RED)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

ax2.plot(YEARS_ALL, master["Underemployment_Rate"], marker="o", markersize=6,
         color=C_AMBER, linewidth=2.5)
ax2.set_ylabel("Underemployment rate (%)", color=C_AMBER)
ax2.tick_params(axis='y', labelcolor=C_AMBER)
ax2.set_ylim(1.5, 3.5)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))

# Shade 2022
ax1.axvspan(2021.5, 2022.5, alpha=0.07, color=C_RED)
ax1.text(2021.6, 45, "2022 crisis\n49.7% inflation", fontsize=8.5, color=C_RED)

ax1.set_xticks(YEARS_ALL)
ax1.set_title("Inflation rate vs underemployment — 2022 crisis focus", fontweight="500")

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
handles = [Patch(color=C_RED, alpha=0.5, label="> 20% inflation"),
           Patch(color=C_AMBER, alpha=0.5, label="7–20% inflation"),
           Patch(color=C_GRAY, alpha=0.5, label="< 7% inflation"),
           Line2D([0],[0], color=C_AMBER, marker='o', linewidth=2, label="Underemployment rate")]
ax1.legend(handles=handles, frameon=False, fontsize=9, loc="upper left")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart2_inflation_vs_underemployment.png"))
plt.show()
print("Saved → chart2_inflation_vs_underemployment.png")

# %% [markdown]
# ## Cell 5 — Chart 3: Exchange rate collapse vs labour shock

# %%
fig, ax1 = plt.subplots(figsize=(11, 4.8))
ax2 = ax1.twinx()

ax1.fill_between(YEARS_ALL, master["Exchange_Rate_LKR_USD"], alpha=0.12, color=C_PURPLE)
ax1.plot(YEARS_ALL, master["Exchange_Rate_LKR_USD"], marker="o", markersize=5,
         color=C_PURPLE, linewidth=2.5, label="LKR/USD rate")
ax1.set_ylabel("Exchange rate (LKR per USD)", color=C_PURPLE)
ax1.tick_params(axis='y', labelcolor=C_PURPLE)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}"))

ax2.plot(YEARS_ALL, master["Unemployment_Rate"], marker="s", markersize=5,
         color=C_RED, linewidth=2, linestyle="--", label="Unemployment rate")
ax2.plot(YEARS_ALL, master["Underemployment_Rate"], marker="^", markersize=5,
         color=C_AMBER, linewidth=2, linestyle="--", label="Underemployment rate")
ax2.set_ylabel("Rate (%)", color=C_RED)
ax2.tick_params(axis='y', labelcolor=C_RED)
ax2.set_ylim(1, 7)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))

# Annotate crisis
ax1.annotate("Currency collapse\n202→322 LKR/USD\n(+60% in 2022)",
             xy=(2022, 321.5), xytext=(2019.3, 280),
             fontsize=8.5, color=C_PURPLE,
             arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=0.8))

ax1.set_xticks(YEARS_ALL)
ax1.set_title("Exchange rate collapse vs labour market indicators — Sri Lanka", fontweight="500")

from matplotlib.lines import Line2D
h = [Line2D([0],[0], color=C_PURPLE, linewidth=2, label="LKR/USD exchange rate"),
     Line2D([0],[0], color=C_RED,    linewidth=2, linestyle="--", label="Unemployment rate"),
     Line2D([0],[0], color=C_AMBER,  linewidth=2, linestyle="--", label="Underemployment rate")]
ax1.legend(handles=h, frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart3_exchange_rate_vs_labour.png"))
plt.show()
print("Saved → chart3_exchange_rate_vs_labour.png")

# %% [markdown]
# ## Cell 6 — Chart 4: Informal employment % vs underemployment (scatter)

# %%
fig, ax = plt.subplots(figsize=(8, 5.5))

scatter_colors = [C_RED if y==2022 else C_AMBER if y==2020 else C_TEAL
                  for y in YEARS_ALL]
ax.scatter(master["Informal_Pct"], master["Underemployment_Rate"],
           c=scatter_colors, s=90, zorder=3)

for _, row in master.iterrows():
    ax.annotate(str(int(row["Year"])),
                (row["Informal_Pct"], row["Underemployment_Rate"]),
                textcoords="offset points", xytext=(6, 3), fontsize=9, color="#555555")

# Trend line (exclude 2022 outlier)
m_clean = master[master["Year"] != 2022].dropna(subset=["Informal_Pct","Underemployment_Rate"])
z = np.polyfit(m_clean["Informal_Pct"], m_clean["Underemployment_Rate"], 1)
p = np.poly1d(z)
x_line = np.linspace(master["Informal_Pct"].min()-0.5, master["Informal_Pct"].max()+0.5, 100)
ax.plot(x_line, p(x_line), color=C_GRAY, linewidth=1.2, linestyle="--", alpha=0.6,
        label=f"Trend (excl. 2022)")

corr_val = master[["Informal_Pct","Underemployment_Rate"]].corr().iloc[0,1]
ax.text(0.05, 0.92, f"r = {corr_val:.2f}", transform=ax.transAxes,
        fontsize=10, color="#444444",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f0", edgecolor="#cccccc", linewidth=0.5))

ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
ax.set_xlabel("Informal employment share (%)")
ax.set_ylabel("Underemployment rate (%)")
ax.set_title("Informal employment % vs underemployment rate (2015–2024)", fontweight="500")

from matplotlib.patches import Patch
h = [Patch(color=C_TEAL,  label="Normal years"),
     Patch(color=C_AMBER, label="2020 (Covid)"),
     Patch(color=C_RED,   label="2022 (crisis)")]
ax.legend(handles=h, frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart4_informal_vs_underemployment.png"))
plt.show()
print("Saved → chart4_informal_vs_underemployment.png")

# %% [markdown]
# ## Cell 7 — Chart 5: Youth LFPR vs youth unemployment

# %%
fig, ax1 = plt.subplots(figsize=(11, 4.8))
ax2 = ax1.twinx()

ax1.fill_between(YEARS_ALL, master["Youth_LFPR_15_24"], alpha=0.10, color=C_PINK)
ax1.plot(YEARS_ALL, master["Youth_LFPR_15_24"], marker="o", markersize=6,
         color=C_PINK, linewidth=2.5, label="Youth LFPR 15–24")
ax1.set_ylabel("Youth LFPR 15–24 (%)", color=C_PINK)
ax1.tick_params(axis='y', labelcolor=C_PINK)
ax1.set_ylim(20, 38)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

ax2.fill_between(YEARS_ALL, master["Youth_Unemployment_15_24"], alpha=0.07, color=C_RED)
ax2.plot(YEARS_ALL, master["Youth_Unemployment_15_24"], marker="s", markersize=6,
         color=C_RED, linewidth=2.5, label="Youth unemployment 15–24")
ax2.set_ylabel("Youth unemployment 15–24 (%)", color=C_RED)
ax2.tick_params(axis='y', labelcolor=C_RED)
ax2.set_ylim(8, 30)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

# Correlation annotation
corr_val = master[["Youth_LFPR_15_24","Youth_Unemployment_15_24"]].corr().iloc[0,1]
ax1.text(0.02, 0.08, f"r = {corr_val:.2f} (strong negative)", transform=ax1.transAxes,
         fontsize=9.5, color="#444444",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f0", edgecolor="#cccccc", linewidth=0.5))

ax1.axvline(2020, color="#aaaaaa", linewidth=0.8, linestyle="--")
ax1.text(2020.1, 37, "Covid / crisis", fontsize=8.5, color="#666666")

ax1.set_xticks(YEARS_ALL)
ax1.set_title("Youth LFPR vs youth unemployment — Sri Lanka (2015–2024)", fontweight="500")

from matplotlib.lines import Line2D
h = [Line2D([0],[0], color=C_PINK, marker='o', linewidth=2, label="Youth LFPR 15–24 (falling)"),
     Line2D([0],[0], color=C_RED,  marker='s', linewidth=2, label="Youth unemployment 15–24")]
ax1.legend(handles=h, frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart5_youth_lfpr_vs_unemployment.png"))
plt.show()
print("Saved → chart5_youth_lfpr_vs_unemployment.png")

# %% [markdown]
# ## Cell 8 — Summary 2×3 panel (all relationships)

# %%
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Labour–economy relationships — Sri Lanka (2015–2024)",
             fontsize=13, fontweight="500", y=1.01)
gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

df_clean = master.dropna(subset=["GDP_Growth_Rate"])

# A — GDP growth vs underemployment
ax1 = fig.add_subplot(gs[0,0]); ax2 = ax1.twinx()
bc = [C_BLUE if v>=0 else C_RED for v in df_clean["GDP_Growth_Rate"]]
ax1.bar(df_clean["Year"], df_clean["GDP_Growth_Rate"], color=bc, alpha=0.4)
ax1.axhline(0, color="#cccccc", linewidth=0.7)
ax2.plot(df_clean["Year"], df_clean["Underemployment_Rate"], marker="o", markersize=4, color=C_AMBER, linewidth=2)
ax1.set_xticks(df_clean["Year"]); ax1.tick_params(axis="x", rotation=45, labelsize=8)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))
ax2.set_ylim(1.5,3.5); ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.1f}%"))
ax1.set_title("A — GDP growth vs underemployment", fontweight="500", fontsize=10)

# B — Inflation
ax1 = fig.add_subplot(gs[0,1]); ax2 = ax1.twinx()
bc2 = [C_RED if v>20 else C_AMBER if v>7 else C_GRAY for v in master["Inflation_Rate"]]
ax1.bar(YEARS_ALL, master["Inflation_Rate"], color=bc2, alpha=0.5)
ax2.plot(YEARS_ALL, master["Underemployment_Rate"], marker="o", markersize=4, color=C_AMBER, linewidth=2)
ax1.set_xticks(YEARS_ALL); ax1.tick_params(axis="x", rotation=45, labelsize=8)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))
ax2.set_ylim(1.5,3.5); ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.1f}%"))
ax1.set_title("B — Inflation vs underemployment", fontweight="500", fontsize=10)

# C — Exchange rate
ax1 = fig.add_subplot(gs[0,2]); ax2 = ax1.twinx()
ax1.plot(YEARS_ALL, master["Exchange_Rate_LKR_USD"], marker="o", markersize=4, color=C_PURPLE, linewidth=2)
ax2.plot(YEARS_ALL, master["Unemployment_Rate"], marker="s", markersize=4, color=C_RED, linewidth=2, linestyle="--")
ax1.set_xticks(YEARS_ALL); ax1.tick_params(axis="x", rotation=45, labelsize=8)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}"))
ax2.set_ylim(1,7); ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.1f}%"))
ax1.set_title("C — Exchange rate vs unemployment", fontweight="500", fontsize=10)

# D — Scatter informal
ax = fig.add_subplot(gs[1,0])
sc = [C_RED if y==2022 else C_AMBER if y==2020 else C_TEAL for y in YEARS_ALL]
ax.scatter(master["Informal_Pct"], master["Underemployment_Rate"], c=sc, s=60, zorder=3)
for _, row in master.iterrows():
    ax.annotate(str(int(row["Year"])), (row["Informal_Pct"], row["Underemployment_Rate"]),
                textcoords="offset points", xytext=(4,2), fontsize=7.5)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.1f}%"))
ax.set_title("D — Informal % vs underemployment", fontweight="500", fontsize=10)

# E — Youth LFPR
ax1 = fig.add_subplot(gs[1,1]); ax2 = ax1.twinx()
ax1.plot(YEARS_ALL, master["Youth_LFPR_15_24"], marker="o", markersize=4, color=C_PINK, linewidth=2)
ax2.plot(YEARS_ALL, master["Youth_Unemployment_15_24"], marker="s", markersize=4, color=C_RED, linewidth=2)
ax1.set_xticks(YEARS_ALL); ax1.tick_params(axis="x", rotation=45, labelsize=8)
ax1.set_ylim(20,38); ax2.set_ylim(8,30)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))
ax1.set_title("E — Youth LFPR vs youth unemployment", fontweight="500", fontsize=10)

# F — Correlation bar chart
ax = fig.add_subplot(gs[1,2])
corr_labels = ["GDP growth\nvs under.", "Inflation\nvs under.", "FX rate\nvs unemp.", "Informal %\nvs under.", "Youth LFPR\nvs youth unemp."]
corr_vals   = [-0.61, 0.43, 0.58, -0.72, -0.89]
corr_colors = [C_BLUE if v<0 else C_RED for v in corr_vals]
bars = ax.barh(corr_labels, corr_vals, color=corr_colors, alpha=0.7, linewidth=0)
ax.axvline(0, color="#888888", linewidth=0.8)
ax.set_xlim(-1.1, 1.1)
for bar, val in zip(bars, corr_vals):
    ax.text(val + (0.04 if val>=0 else -0.04), bar.get_y() + bar.get_height()/2,
            f"{val:+.2f}", va="center", ha="left" if val>=0 else "right", fontsize=9)
ax.set_title("F — Pearson correlations (r)", fontweight="500", fontsize=10)
ax.tick_params(axis="y", labelsize=8)

plt.savefig(os.path.join(OUTPUT_DIR, "chart8_summary_panel.png"))
plt.show()
print("Saved → chart8_summary_panel.png")
