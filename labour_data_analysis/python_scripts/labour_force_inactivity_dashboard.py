# %% [markdown]
# # Labour Force & Inactivity Dashboard — Sri Lanka (2015–2024)
# ### Source: DCS Labour Force Survey + ILO ILOSTAT (TRU series)
# Run each cell independently or top-to-bottom.

# %% [markdown]
# ## Cell 0 — Imports & shared style

# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#e5e5e5",
    "grid.linewidth":    0.6,
    "axes.axisbelow":    True,
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
})

C_AMBER  = "#BA7517"
C_BLUE   = "#185FA5"
C_GRAY   = "#888780"
C_PINK   = "#D4537E"
C_TEAL   = "#1D9E75"
C_GREEN  = "#3B6D11"
C_RED    = "#E24B4A"

OUTPUT_DIR = "charts_labour_force"
os.makedirs(OUTPUT_DIR, exist_ok=True)
YEARS = list(range(2015, 2025))
print("Style loaded.")


# %% [markdown]
# ## Cell 1 — Load CSV files

# %%
DATA_DIR = "."   # change to folder containing your CSVs

df_tru_f    = pd.read_csv(os.path.join(DATA_DIR, "tru-female.csv"))
df_tru_m    = pd.read_csv(os.path.join(DATA_DIR, "tru-male.csv"))
df_reasons  = pd.read_csv(os.path.join(DATA_DIR, "Table_3_9_-_Reasons_for_being_economically_inactive_by_gender.csv"))
df_pot      = pd.read_csv(os.path.join(DATA_DIR, "Table_3_12_-_Potential_labour_force_by_year_and_gender.csv"))
df_disc     = pd.read_csv(os.path.join(DATA_DIR, "Table_3_13_-_Discouraged_Job_Seekers_by_gender.csv"))

# Fix potential labour force columns (mixed format in source)
# Some rows have (Number, Rate, Gender) and some have (Year, Number/Gender, Rate)
# We'll parse carefully
print("Files loaded.")
for name, df in [("df_tru_f",df_tru_f),("df_tru_m",df_tru_m),("df_reasons",df_reasons),
                 ("df_pot",df_pot),("df_disc",df_disc)]:
    print(f"  {name}: {df.shape}")


# %% [markdown]
# ## Cell 2 — Prepare TRU series (2010–2023)

# %%
tru_f = df_tru_f[["Year","Value"]].rename(columns={"Value":"Female_TRU"}).sort_values("Year")
tru_m = df_tru_m[["Year","Value"]].rename(columns={"Value":"Male_TRU"}).sort_values("Year")
tru   = pd.merge(tru_f, tru_m, on="Year")

# Filter to study window + a few years before for context
tru_plot = tru[tru["Year"] >= 2010].copy()
print(tru_plot.to_string(index=False))


# %% [markdown]
# ## Cell 3 — Chart 1: Time-related underemployment by gender (ILO, 2010–2023)

# %%
fig, ax = plt.subplots(figsize=(10, 4.5))

ax.fill_between(tru_plot["Year"], tru_plot["Female_TRU"], alpha=0.12, color=C_PINK)
ax.fill_between(tru_plot["Year"], tru_plot["Male_TRU"],   alpha=0.08, color=C_BLUE)

ax.plot(tru_plot["Year"], tru_plot["Female_TRU"],
        marker="o", markersize=5, color=C_PINK, linewidth=2, label="Female TRU %")
ax.plot(tru_plot["Year"], tru_plot["Male_TRU"],
        marker="o", markersize=5, color=C_BLUE, linewidth=2, label="Male TRU %")

# Shade 2022 crisis area
ax.axvspan(2021.5, 2022.5, alpha=0.06, color="red")
ax.text(2021.6, 4.3, "2022\ncrisis", fontsize=8.5, color="#b00000")

ax.set_xticks(tru_plot["Year"].tolist())
ax.tick_params(axis="x", labelrotation=45)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
ax.set_ylabel("Time-related underemployment (% of employment)")
ax.set_title("Time-related underemployment by gender — Sri Lanka (ILO, 2010–2023)", fontweight="500")
ax.legend(frameon=False, fontsize=10)

# Gender gap annotation
gap_2023 = tru_plot[tru_plot["Year"]==2023]["Female_TRU"].values[0] - tru_plot[tru_plot["Year"]==2023]["Male_TRU"].values[0]
ax.annotate(f"Gap = {gap_2023:.2f}pp\nin 2023",
            xy=(2023, (tru_plot[tru_plot["Year"]==2023]["Female_TRU"].values[0] + tru_plot[tru_plot["Year"]==2023]["Male_TRU"].values[0])/2),
            xytext=(2021, 3.9), fontsize=8.5, color="#444444",
            arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart1_tru_gender.png"))
plt.show()
print("Saved → charts_labour_force/chart1_tru_gender.png")


# %% [markdown]
# ## Cell 4 — Chart 2: Reasons for economic inactivity — female (2015–2024)

# %%
reasons_order = ["Engaged in housework", "Engaged in studies",
                 "Retired/Old age", "Physically illness/Disabled", "Other"]
colors_r = [C_PINK, C_BLUE, C_GRAY, C_AMBER, C_TEAL]
markers_r = ["o", "s", "D", "^", "x"]

female_reasons = df_reasons[
    (df_reasons["Reason"] != "All Economically Inactive")
].sort_values("Year")

fig, ax = plt.subplots(figsize=(10, 4.5))

for reason, col, mk in zip(reasons_order, colors_r, markers_r):
    sub = female_reasons[female_reasons["Reason"] == reason]
    if not sub.empty:
        ax.plot(sub["Year"], sub["Female_(%)"],
                marker=mk, markersize=5, color=col, linewidth=2, label=reason)

ax.set_xticks(YEARS)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_ylabel("% of economically inactive females")
ax.set_title("Reasons for economic inactivity — female (2015–2024)", fontweight="500")
ax.legend(frameon=False, fontsize=9, loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart2_inactivity_reasons_female.png"))
plt.show()
print("Saved → charts_labour_force/chart2_inactivity_reasons_female.png")


# %% [markdown]
# ## Cell 5 — Chart 3: Male vs female inactivity reasons (2024)

# %%
yr2024 = df_reasons[
    (df_reasons["Year"] == 2024) &
    (df_reasons["Reason"] != "All Economically Inactive")
].copy()

reasons_short = {
    "Engaged in housework": "Housework",
    "Engaged in studies": "Studies",
    "Retired/Old age": "Retired/old age",
    "Physically illness/Disabled": "Illness/disability",
    "Other": "Other"
}
yr2024["Reason_Short"] = yr2024["Reason"].map(reasons_short)

x = np.arange(len(yr2024))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 4.5))
b1 = ax.bar(x - width/2, yr2024["Male_(%)"],   width, color=C_BLUE, label="Male %",   linewidth=0, zorder=3)
b2 = ax.bar(x + width/2, yr2024["Female_(%)"], width, color=C_PINK, label="Female %", linewidth=0, zorder=3)

for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=8.5, color="#444444")

ax.set_xticks(x)
ax.set_xticklabels(yr2024["Reason_Short"], fontsize=10)
ax.set_ylim(0, 70)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_ylabel("% of economically inactive")
ax.set_title("Inactivity reasons — male vs female (2024)", fontweight="500")
ax.legend(frameon=False, fontsize=10)

# Housework annotation
ax.annotate("Housework trap:\n58.6% of inactive\nwomen cite housework",
            xy=(0 + width/2, yr2024[yr2024["Reason_Short"]=="Housework"]["Female_(%)"].values[0]),
            xytext=(1.2, 62), fontsize=8.5, color=C_PINK,
            arrowprops=dict(arrowstyle="->", color=C_PINK, lw=0.8))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart3_inactivity_comparison.png"))
plt.show()
print("Saved → charts_labour_force/chart3_inactivity_comparison.png")


# %% [markdown]
# ## Cell 6 — Chart 4: Discouraged job seekers (2017–2024)

# %%
disc_total  = df_disc[df_disc["Gender"] == "Total"].sort_values("Year")
disc_male   = df_disc[df_disc["Gender"] == "Male"].sort_values("Year")
disc_female = df_disc[df_disc["Gender"] == "Female"].sort_values("Year")

fig, ax = plt.subplots(figsize=(10, 4.2))

ax.bar(disc_female["Year"] - 0.2, disc_female["Number"], 0.4,
       color=C_PINK, label="Female", linewidth=0, zorder=3)
ax.bar(disc_male["Year"]   + 0.2, disc_male["Number"],   0.4,
       color=C_BLUE, label="Male",   linewidth=0, zorder=3)

ax.plot(disc_total["Year"], disc_total["Number"],
        marker="D", markersize=5, color=C_GRAY, linewidth=1.5,
        linestyle="--", label="Total", zorder=4)

ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
ax.set_ylabel("Number of discouraged job seekers")
ax.set_xticks(disc_total["Year"].tolist())
ax.set_title("Discouraged job seekers by gender — Sri Lanka (2017–2024)", fontweight="500")
ax.legend(frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart4_discouraged.png"))
plt.show()
print("Saved → charts_labour_force/chart4_discouraged.png")


# %% [markdown]
# ## Cell 7 — Chart 5: Potential labour force as % of inactive (2015–2024)

# %%
# Standardise the messy potential labour force CSV
# Some rows: (Number, Rate_to_Inactive_%, Gender) — early years
# Some rows: (Year, Number_or_Gender, Rate_or_Number) — later years
# We'll rebuild cleanly from raw values we know

pot_data = {
    "Year":   [2015,2016,2017,2018,2019,2020,2021,2022,2023,2024],
    "Total":  [3.1, 2.9, 3.2, 2.6, 2.6, 3.0, 2.7, 2.1, 2.7, 1.9],
    "Male":   [4.1, 4.5, 3.9, 3.8, 3.3, 4.2, 4.1, 2.6, 3.7, 2.5],
    "Female": [2.7, 2.4, 2.9, 2.1, 2.3, 2.6, 2.1, 2.0, 2.4, 1.7],
}
pot_df = pd.DataFrame(pot_data)

fig, ax = plt.subplots(figsize=(10, 4.2))

ax.plot(pot_df["Year"], pot_df["Male"],   marker="o", markersize=5, color=C_BLUE, linewidth=2, label="Male")
ax.plot(pot_df["Year"], pot_df["Female"], marker="o", markersize=5, color=C_PINK, linewidth=2, label="Female")
ax.plot(pot_df["Year"], pot_df["Total"],  marker="D", markersize=4, color=C_GRAY, linewidth=1.5, linestyle="-.", label="Total")

ax.set_xticks(YEARS)
ax.set_ylim(0, 6)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
ax.set_ylabel("% of economically inactive population")
ax.set_title("Potential labour force as % of inactive — Sri Lanka (2015–2024)", fontweight="500")
ax.legend(frameon=False, fontsize=10)
ax.annotate("Declining trend:\nfewer people available\nto enter labour market",
            xy=(2024, 1.9), xytext=(2021.5, 4.5), fontsize=8.5, color="#444444",
            arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart5_potential_lf.png"))
plt.show()
print("Saved → charts_labour_force/chart5_potential_lf.png")


# %% [markdown]
# ## Cell 8 — Chart 6: TRU vs LFS underemployment rate (2015–2023 overlap)

# %%
# LFS composite underemployment (from earlier analysis)
lfs_under = pd.DataFrame({
    "Year": [2015,2016,2017,2018,2019,2020,2021,2022,2023],
    "LFS_Total": [2.7, 2.4, 2.8, 2.6, 2.7, 2.6, 2.5, 2.3, 2.5],
})

tru_overlap = tru[tru["Year"].between(2015, 2023)].copy()

fig, ax = plt.subplots(figsize=(10, 4.5))

ax.plot(lfs_under["Year"], lfs_under["LFS_Total"],
        marker="o", markersize=5, color=C_AMBER, linewidth=2, label="LFS composite underemployment (DCS)")
ax.plot(tru_overlap["Year"], tru_overlap["Female_TRU"],
        marker="s", markersize=5, color=C_PINK,  linewidth=2, linestyle="--", label="ILO TRU — female")
ax.plot(tru_overlap["Year"], tru_overlap["Male_TRU"],
        marker="s", markersize=5, color=C_BLUE,  linewidth=2, linestyle="--", label="ILO TRU — male")

ax.set_xticks(list(range(2015, 2024)))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
ax.set_ylabel("Rate (%)")
ax.set_title("LFS composite vs ILO time-related underemployment (2015–2023)", fontweight="500")
ax.legend(frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart6_lfs_vs_tru.png"))
plt.show()
print("Saved → charts_labour_force/chart6_lfs_vs_tru.png")


# %% [markdown]
# ## Cell 9 — Summary 2×2 panel

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Sri Lanka labour force & inactivity analysis — key panels",
             fontsize=13, fontweight="500", y=1.01)

# A — TRU
ax = axes[0, 0]
ax.fill_between(tru_plot["Year"], tru_plot["Female_TRU"], alpha=0.12, color=C_PINK)
ax.fill_between(tru_plot["Year"], tru_plot["Male_TRU"],   alpha=0.08, color=C_BLUE)
ax.plot(tru_plot["Year"], tru_plot["Female_TRU"], marker="o", markersize=4, color=C_PINK, linewidth=2, label="Female TRU")
ax.plot(tru_plot["Year"], tru_plot["Male_TRU"],   marker="o", markersize=4, color=C_BLUE, linewidth=2, label="Male TRU")
ax.set_xticks(tru_plot["Year"].tolist()); ax.tick_params(axis="x", labelrotation=45, labelsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
ax.set_title("A — Time-related underemployment", fontweight="500", fontsize=10)
ax.legend(frameon=False, fontsize=9)

# B — Inactivity reasons female
ax = axes[0, 1]
for reason, col, mk in zip(reasons_order, colors_r, markers_r):
    sub = female_reasons[female_reasons["Reason"] == reason]
    if not sub.empty:
        ax.plot(sub["Year"], sub["Female_(%)"], marker=mk, markersize=4, color=col, linewidth=2, label=reason[:15])
ax.set_xticks(YEARS); ax.tick_params(axis="x", labelrotation=45, labelsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_title("B — Female inactivity reasons", fontweight="500", fontsize=10)
ax.legend(frameon=False, fontsize=8)

# C — Discouraged seekers
ax = axes[1, 0]
ax.bar(disc_female["Year"] - 0.2, disc_female["Number"], 0.4, color=C_PINK, label="Female", linewidth=0)
ax.bar(disc_male["Year"]   + 0.2, disc_male["Number"],   0.4, color=C_BLUE, label="Male",   linewidth=0)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
ax.set_xticks(disc_total["Year"].tolist()); ax.tick_params(axis="x", labelrotation=45, labelsize=8)
ax.set_title("C — Discouraged job seekers", fontweight="500", fontsize=10)
ax.legend(frameon=False, fontsize=9)

# D — Potential labour force
ax = axes[1, 1]
ax.plot(pot_df["Year"], pot_df["Male"],   marker="o", markersize=4, color=C_BLUE, linewidth=2, label="Male")
ax.plot(pot_df["Year"], pot_df["Female"], marker="o", markersize=4, color=C_PINK, linewidth=2, label="Female")
ax.plot(pot_df["Year"], pot_df["Total"],  marker="D", markersize=3, color=C_GRAY, linewidth=1.5, linestyle="-.", label="Total")
ax.set_xticks(YEARS); ax.tick_params(axis="x", labelrotation=45, labelsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
ax.set_title("D — Potential labour force (% inactive)", fontweight="500", fontsize=10)
ax.legend(frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart9_summary_panel.png"))
plt.show()
print("Saved → charts_labour_force/chart9_summary_panel.png")
