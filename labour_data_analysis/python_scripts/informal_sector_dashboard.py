# %% [markdown]
# # Informal Sector Dashboard — Sri Lanka (2015–2024)
# ### Source: Department of Census and Statistics, Labour Force Survey
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

C_RED    = "#E24B4A"
C_AMBER  = "#BA7517"
C_BLUE   = "#185FA5"
C_GREEN  = "#3B6D11"
C_GRAY   = "#888780"
C_PINK   = "#D4537E"
C_TEAL   = "#1D9E75"

OUTPUT_DIR = "charts_informal"
os.makedirs(OUTPUT_DIR, exist_ok=True)
YEARS = list(range(2015, 2025))
print("Style loaded.")


# %% [markdown]
# ## Cell 1 — Load CSV files

# %%
DATA_DIR = "."   # change to folder containing your CSVs

df_sector   = pd.read_csv(os.path.join(DATA_DIR, "Distribution_of_Informal_Formal_sector_employment_by_economic_sector.csv"))
df_gender   = pd.read_csv(os.path.join(DATA_DIR, "Distribution_of_Informal_Formal_sector_employment_by_gender.csv"))
df_edu      = pd.read_csv(os.path.join(DATA_DIR, "Distribution_of_Informal_Formal_sector_employment_by_level_of_education.csv"))
df_status   = pd.read_csv(os.path.join(DATA_DIR, "Distribution_of_Informal_Formal_sector_employment_by_employment_status.csv"))
df_occ      = pd.read_csv(os.path.join(DATA_DIR, "Distribution_of_Informal_Formal_sector_employment_by_main_occupation.csv"))
df_industry = pd.read_csv(os.path.join(DATA_DIR, "Distribution_of_Informal_sector_employment_by_major_industry_group.csv"))
df_highlights = pd.read_csv(os.path.join(DATA_DIR, "Highlights_-_Employment_Contribution_to_Informal_Sector.csv"))

print("Files loaded.")
for name, df in [("df_sector",df_sector),("df_gender",df_gender),("df_edu",df_edu),
                 ("df_status",df_status),("df_occ",df_occ),("df_industry",df_industry),
                 ("df_highlights",df_highlights)]:
    print(f"  {name}: {df.shape}")


# %% [markdown]
# ## Cell 2 — Chart 1: Informal share — total, agriculture, non-agriculture (2015–2024)

# %%
hl = df_highlights.sort_values("Year")

fig, ax = plt.subplots(figsize=(10, 4.5))

ax.fill_between(hl["Year"], hl["Agriculture_Pct"],     alpha=0.10, color=C_GREEN)
ax.fill_between(hl["Year"], hl["Non_Agriculture_Pct"], alpha=0.08, color=C_BLUE)

ax.plot(hl["Year"], hl["Agriculture_Pct"],     marker="o", markersize=5, color=C_GREEN, linewidth=2, label="Agriculture")
ax.plot(hl["Year"], hl["Sri_Lanka_Total_Pct"], marker="s", markersize=5, color=C_AMBER, linewidth=2, label="Total")
ax.plot(hl["Year"], hl["Non_Agriculture_Pct"], marker="^", markersize=5, color=C_BLUE,  linewidth=2, linestyle="--", label="Non-agriculture")

ax.set_ylim(40, 95)
ax.set_xticks(YEARS)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_ylabel("Informal employment share (%)")
ax.set_title("Informal employment share: total, agriculture, non-agriculture (2015–2024)", fontweight="500")
ax.legend(frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart1_informal_trend.png"))
plt.show()
print("Saved → charts_informal/chart1_informal_trend.png")


# %% [markdown]
# ## Cell 3 — Chart 2: Informal employment % by industry (2015 vs 2024)

# %%
ind_2015 = df_industry[df_industry["Year"] == 2015].set_index("Industry_Group")["Informal_Pct"]
ind_2024 = df_industry[df_industry["Year"] == 2024].set_index("Industry_Group")["Informal_Pct"]
common = ind_2015.index.intersection(ind_2024.index)

# Short labels
short_labels = {
    "Agriculture Forestry and Fishing": "Agri/Forest/Fish",
    "Construction Electricity Gas Steam Water Supply Sewerage Waste Management": "Construction\n& Utilities",
    "Accommodation and Food Services": "Accommodation\n& Food",
    "Manufacturing": "Manufacturing",
    "Education": "Education",
}
labels = [short_labels.get(i, i) for i in common]

x = np.arange(len(common))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 4.5))
b1 = ax.bar(x - width/2, ind_2015[common].values, width, color=C_BLUE,  label="2015", linewidth=0, zorder=3)
b2 = ax.bar(x + width/2, ind_2024[common].values, width, color=C_AMBER, label="2024", linewidth=0, zorder=3)

for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=8.5, color="#444444")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_ylabel("Informal employment (%)")
ax.set_title("Informal employment % by industry — 2015 vs 2024", fontweight="500")
ax.legend(frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart2_industry_comparison.png"))
plt.show()
print("Saved → charts_informal/chart2_industry_comparison.png")


# %% [markdown]
# ## Cell 4 — Chart 3: Informal employment % by education level (2015 vs 2024)

# %%
def edu_informal(df, year):
    sub = df[df["Year"] == year].copy()
    sub = sub[sub["Level_of_Education"] != "Total"]
    return sub.set_index("Level_of_Education")["Informal_Pct"]

# Harmonise education labels across years
edu_map = {
    "Below Grade 6": "Below Grade 6",
    "Grade 5 and below": "Below Grade 6",
    "Grade 6-10": "Grade 6–10",
    "GCE O/L": "GCE O/L",
    "GCE A/L and above": "GCE A/L+",
}

def get_edu_series(df, year):
    sub = df[df["Year"] == year].copy()
    sub = sub[sub["Level_of_Education"] != "Total"]
    sub["Level_of_Education"] = sub["Level_of_Education"].map(edu_map)
    sub = sub.dropna(subset=["Level_of_Education"])
    return sub.groupby("Level_of_Education")["Informal_Pct"].first()

edu_order = ["Below Grade 6", "Grade 6–10", "GCE O/L", "GCE A/L+"]
e2015 = get_edu_series(df_edu, 2015).reindex(edu_order)
e2024 = get_edu_series(df_edu, 2024).reindex(edu_order)

x = np.arange(len(edu_order))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 4.5))
b1 = ax.bar(x - width/2, e2015.values, width, color=C_BLUE,  label="2015", linewidth=0, zorder=3)
b2 = ax.bar(x + width/2, e2024.values, width, color=C_AMBER, label="2024", linewidth=0, zorder=3)

for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.0f}%",
                    ha="center", va="bottom", fontsize=8.5, color="#444444")

ax.set_xticks(x)
ax.set_xticklabels(edu_order, fontsize=10)
ax.set_ylim(0, 92)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_ylabel("Informal employment (%)")
ax.set_title("Informal employment % by education level — 2015 vs 2024", fontweight="500")
ax.legend(frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart3_education.png"))
plt.show()
print("Saved → charts_informal/chart3_education.png")


# %% [markdown]
# ## Cell 5 — Chart 4: Informal % by employment status (2024)

# %%
status_2024 = df_status[(df_status["Year"] == 2024) & (df_status["Employment_Status"] != "Total")].copy()
status_2024 = status_2024.sort_values("Informal_Pct", ascending=True)

colors_s = [C_RED if v > 50 else C_AMBER if v > 30 else C_BLUE
            for v in status_2024["Informal_Pct"]]

fig, ax = plt.subplots(figsize=(9, 3.5))
bars = ax.barh(status_2024["Employment_Status"], status_2024["Informal_Pct"],
               color=colors_s, linewidth=0, zorder=3)

for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}%",
            va="center", fontsize=9, color="#444444")

ax.set_xlim(0, 65)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_xlabel("Informal employment (%)")
ax.set_title("Informal employment % by employment status — 2024", fontweight="500")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart4_employment_status.png"))
plt.show()
print("Saved → charts_informal/chart4_employment_status.png")


# %% [markdown]
# ## Cell 6 — Chart 5: Informal % by occupation (2024) — sorted

# %%
occ_2024 = df_occ[(df_occ["Year"] == 2024) & (df_occ["Occupation"] != "Total")].copy()
occ_2024 = occ_2024.sort_values("Informal_Pct", ascending=True)

# Shorten occupation labels
occ_short = {
    "Skilled Agricultural / Forestry & Fishery Workers": "Skilled agri/forest/fish",
    "Craft & Related Trades Workers": "Craft & trades",
    "Elementary Occupations": "Elementary occupations",
    "Plant & Machine Operators & Assemblers": "Plant & machine operators",
    "Hospitality / Shop & Related Services Managers": "Hospitality/shop mgrs",
    "Services & Sales Workers": "Services & sales",
    "Professionals": "Professionals",
    "Technical & Associate Professionals": "Tech & assoc. professionals",
    "Production & Specialized Services Managers": "Production managers",
    "Administrative & Commercial Managers": "Administrative managers",
    "Chief Executive / Senior Official / Legislators": "Chief executives",
    "Armed Forces & Unidentified Occupations": "Armed forces",
}
occ_2024["Short_Label"] = occ_2024["Occupation"].map(occ_short).fillna(occ_2024["Occupation"])

colors_occ = [C_RED if v > 80 else C_AMBER if v > 50 else C_BLUE
              for v in occ_2024["Informal_Pct"]]

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(occ_2024["Short_Label"], occ_2024["Informal_Pct"],
        color=colors_occ, linewidth=0, zorder=3)

ax.axvline(50, color="#aaaaaa", linewidth=0.8, linestyle="--")
ax.text(50.5, len(occ_2024) - 1, "50% threshold", fontsize=8.5, color="#666666")

ax.set_xlim(0, 105)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_xlabel("Informal employment (%)")
ax.set_title("Informal employment % by occupation — 2024 (sorted)", fontweight="500")

from matplotlib.patches import Patch
legend_els = [Patch(facecolor=C_RED,   label="> 80%"),
              Patch(facecolor=C_AMBER, label="50–80%"),
              Patch(facecolor=C_BLUE,  label="< 50%")]
ax.legend(handles=legend_els, frameon=False, fontsize=9, loc="lower right")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart5_occupation.png"))
plt.show()
print("Saved → charts_informal/chart5_occupation.png")


# %% [markdown]
# ## Cell 7 — Chart 6: Gender informal trend (2015–2024)

# %%
inf_male = df_gender[df_gender["Gender"] == "Male"].sort_values("Year")
inf_fem  = df_gender[df_gender["Gender"] == "Female"].sort_values("Year")

fig, ax = plt.subplots(figsize=(10, 4.2))

ax.fill_between(inf_male["Year"], inf_male["Informal_Pct"], alpha=0.10, color=C_BLUE)
ax.fill_between(inf_fem["Year"],  inf_fem["Informal_Pct"],  alpha=0.10, color=C_TEAL)

ax.plot(inf_male["Year"], inf_male["Informal_Pct"],
        marker="o", markersize=5, color=C_BLUE,  linewidth=2, label="Male informal %")
ax.plot(inf_fem["Year"],  inf_fem["Informal_Pct"],
        marker="o", markersize=5, color=C_TEAL,  linewidth=2, label="Female informal %")

# Annotate key years
for yr in [2015, 2019, 2022, 2024]:
    m = inf_male[inf_male["Year"] == yr]["Informal_Pct"].values
    f = inf_fem[inf_fem["Year"] == yr]["Informal_Pct"].values
    if len(m): ax.annotate(f"{m[0]:.1f}%", (yr, m[0]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color=C_BLUE)
    if len(f): ax.annotate(f"{f[0]:.1f}%", (yr, f[0]), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=8, color=C_TEAL)

ax.set_xticks(YEARS)
ax.set_ylim(40, 70)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_ylabel("Informal employment share (%)")
ax.set_title("Informal employment share by gender — Sri Lanka (2015–2024)", fontweight="500")
ax.legend(frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart6_gender_trend.png"))
plt.show()
print("Saved → charts_informal/chart6_gender_trend.png")


# %% [markdown]
# ## Cell 8 — Summary 2×2 panel

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Sri Lanka informal sector analysis — key panels (2015–2024)",
             fontsize=13, fontweight="500", y=1.01)

# A — Trend
ax = axes[0, 0]
ax.plot(hl["Year"], hl["Agriculture_Pct"],     marker="o", markersize=4, color=C_GREEN, linewidth=2, label="Agriculture")
ax.plot(hl["Year"], hl["Sri_Lanka_Total_Pct"], marker="s", markersize=4, color=C_AMBER, linewidth=2, label="Total")
ax.plot(hl["Year"], hl["Non_Agriculture_Pct"], marker="^", markersize=4, color=C_BLUE,  linewidth=2, linestyle="--", label="Non-agri")
ax.set_ylim(40, 95); ax.set_xticks(YEARS); ax.tick_params(axis="x", labelrotation=45, labelsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_title("A — Informal share by sector", fontweight="500", fontsize=10)
ax.legend(frameon=False, fontsize=9)

# B — Industry
ax = axes[0, 1]
x = np.arange(len(common))
ax.bar(x - 0.2, ind_2015[common].values, 0.35, color=C_BLUE,  label="2015", linewidth=0)
ax.bar(x + 0.2, ind_2024[common].values, 0.35, color=C_AMBER, label="2024", linewidth=0)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_title("B — By industry (2015 vs 2024)", fontweight="500", fontsize=10)
ax.legend(frameon=False, fontsize=9)

# C — Education
ax = axes[1, 0]
ax.bar(x - 0.2, e2015.values, 0.35, color=C_BLUE,  label="2015", linewidth=0)
ax.bar(x + 0.2, e2024.values, 0.35, color=C_AMBER, label="2024", linewidth=0)
ax.set_xticks(np.arange(len(edu_order))); ax.set_xticklabels(edu_order, fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_title("C — By education (2015 vs 2024)", fontweight="500", fontsize=10)
ax.legend(frameon=False, fontsize=9)

# D — Gender trend
ax = axes[1, 1]
ax.plot(inf_male["Year"], inf_male["Informal_Pct"], marker="o", markersize=4, color=C_BLUE, linewidth=2, label="Male")
ax.plot(inf_fem["Year"],  inf_fem["Informal_Pct"],  marker="o", markersize=4, color=C_TEAL, linewidth=2, label="Female")
ax.set_ylim(40, 70); ax.set_xticks(YEARS); ax.tick_params(axis="x", labelrotation=45, labelsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_title("D — Gender trend", fontweight="500", fontsize=10)
ax.legend(frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart8_summary_panel.png"))
plt.show()
print("Saved → charts_informal/chart8_summary_panel.png")
