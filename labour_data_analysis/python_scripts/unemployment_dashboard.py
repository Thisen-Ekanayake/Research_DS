# %% [markdown]
# # Unemployment Dashboard — Sri Lanka (2015–2024)
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
C_PURPLE = "#7F77DD"
C_GRAY   = "#888780"
C_PINK   = "#D4537E"

OUTPUT_DIR = "charts_unemployment"
os.makedirs(OUTPUT_DIR, exist_ok=True)
YEARS = list(range(2015, 2025))
print("Style loaded.")


# %% [markdown]
# ## Cell 1 — Load CSV files

# %%
DATA_DIR = "."   # change to folder containing your CSVs

df_age   = pd.read_csv(os.path.join(DATA_DIR, "Table_5_4_Unemployment_rate_by_age_group_and_gender.csv"))
df_edu   = pd.read_csv(os.path.join(DATA_DIR, "Table_5_5_Unemployment_rate_by_level_of_education.csv"))
df_neet  = pd.read_csv(os.path.join(DATA_DIR, "Table_5_12_5_13_NEET_Youth_Not_in_Employment_Education_or_Training.csv"))
df_neet.columns = df_neet.columns.str.strip()
df_neet["Gender"] = df_neet["Gender"].str.strip()
df_dist  = pd.read_csv(os.path.join(DATA_DIR, "Table_5_3_Unemployment_rate_by_district.csv"))

print("Files loaded.")
for name, df in [("df_age",df_age),("df_edu",df_edu),("df_neet",df_neet),("df_dist",df_dist)]:
    print(f"  {name}: {df.shape}")


# %% [markdown]
# ## Cell 2 — Chart 1: Unemployment by age group (2015–2024)

# %%
age_groups = ["15-24", "25-29", "30-39", "40+"]
colors     = [C_RED, C_AMBER, C_BLUE, C_GRAY]
markers    = ["o", "s", "^", "D"]
styles     = ["-", "-", "-", "-."]

fig, ax = plt.subplots(figsize=(10, 4.5))

for ag, col, mk, ls in zip(age_groups, colors, markers, styles):
    sub = df_age[df_age["Age_Group"] == ag].sort_values("Year")
    ax.plot(sub["Year"], sub["Total"],
            marker=mk, markersize=5, color=col, linewidth=2,
            linestyle=ls, label=ag)

ax.axvline(2020, color="#aaaaaa", linewidth=0.8, linestyle="--")
ax.text(2020.1, 27, "Covid / crisis", fontsize=9, color="#666666")

ax.set_xticks(YEARS)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
ax.set_ylabel("Unemployment rate (%)")
ax.set_title("Unemployment rate by age group — Sri Lanka (2015–2024)", fontweight="500")
ax.legend(frameon=False, fontsize=10, title="Age group", title_fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart1_age_group.png"))
plt.show()
print("Saved → charts_unemployment/chart1_age_group.png")


# %% [markdown]
# ## Cell 3 — Chart 2: Unemployment by education level (2015–2024)

# %%
edu_order  = ["G.C.E. (A/L) & above", "G.C.E. (O/L)", "Grade 6-10", "Grade 5 & Below"]
edu_labels = ["A/L & above", "O/L", "Grade 6–10", "Grade 5 & below"]
colors_edu = [C_PURPLE, C_BLUE, C_AMBER, C_GRAY]
dashes     = [None, None, (5,3), (3,3)]

fig, ax = plt.subplots(figsize=(10, 4.5))

for edu, label, col, dash in zip(edu_order, edu_labels, colors_edu, dashes):
    sub = df_edu[df_edu["Education_Level"] == edu].sort_values("Year")
    ls = "--" if dash else "-"
    ax.plot(sub["Year"], sub["Total_Rate"],
            marker="o", markersize=5, color=col, linewidth=2,
            linestyle=ls, label=label)

ax.axvline(2020, color="#aaaaaa", linewidth=0.8, linestyle="--")
ax.text(2020.1, 10.2, "Covid / crisis", fontsize=9, color="#666666")

ax.set_xticks(YEARS)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
ax.set_ylabel("Unemployment rate (%)")
ax.set_title("Unemployment by education level — Sri Lanka (2015–2024)", fontweight="500")
ax.legend(frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart2_education.png"))
plt.show()
print("Saved → charts_unemployment/chart2_education.png")


# %% [markdown]
# ## Cell 4 — Chart 3: NEET rate by gender (2015–2024)

# %%
# Keep the larger survey series (first set of rows per year) for Total/Male/Female
neet_clean = (df_neet[df_neet["NEET_Rate"] > 20]
              .sort_values("Year")
              .drop_duplicates(subset=["Year", "Gender"], keep="first"))

neet_total  = neet_clean[neet_clean["Gender"] == "Total"]
neet_male   = neet_clean[neet_clean["Gender"] == "Male"]
neet_female = neet_clean[neet_clean["Gender"] == "Female"]

fig, ax = plt.subplots(figsize=(10, 4.5))

ax.fill_between(neet_female["Year"], neet_female["NEET_Rate"], alpha=0.10, color=C_PINK)
ax.fill_between(neet_male["Year"],   neet_male["NEET_Rate"],   alpha=0.08, color=C_BLUE)

ax.plot(neet_female["Year"], neet_female["NEET_Rate"],
        marker="o", markersize=5, color=C_PINK, linewidth=2, label="Female NEET rate")
ax.plot(neet_male["Year"],   neet_male["NEET_Rate"],
        marker="o", markersize=5, color=C_BLUE, linewidth=2, label="Male NEET rate")
ax.plot(neet_total["Year"],  neet_total["NEET_Rate"],
        marker="D", markersize=4, color=C_GRAY, linewidth=1.5, linestyle="-.", label="Total NEET rate")

ax.axvline(2020, color="#aaaaaa", linewidth=0.8, linestyle="--")
ax.text(2020.1, 36.5, "Covid / crisis", fontsize=9, color="#666666")

ax.set_xticks(sorted(neet_total["Year"].unique()))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
ax.set_ylabel("NEET rate (%)")
ax.set_title("NEET rate by gender — Sri Lanka (2015–2024)", fontweight="500")
ax.legend(frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart3_neet_gender.png"))
plt.show()
print("Saved → charts_unemployment/chart3_neet_gender.png")


# %% [markdown]
# ## Cell 5 — Chart 4: Education paradox — unemployment by education & gender (2024)

# %%
edu_2024 = df_edu[df_edu["Year"] == 2024].copy()
edu_order_short = ["Grade 5 & Below", "Grade 6-10", "G.C.E. (O/L)", "G.C.E. (A/L) & above"]
edu_labels_short = ["Grade 5\n& below", "Grade 6–10", "O/L", "A/L\n& above"]

edu_2024 = edu_2024[edu_2024["Education_Level"].isin(edu_order_short)]
edu_2024["Education_Level"] = pd.Categorical(
    edu_2024["Education_Level"], categories=edu_order_short, ordered=True)
edu_2024 = edu_2024.sort_values("Education_Level")

x = np.arange(len(edu_order_short))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 4.5))

bars_t = ax.bar(x - width, edu_2024["Total_Rate"], width, color=C_GRAY,  label="Total",  zorder=3, linewidth=0)
bars_m = ax.bar(x,          edu_2024["Male"],       width, color=C_BLUE,  label="Male",   zorder=3, linewidth=0)
bars_f = ax.bar(x + width,  edu_2024["Female"],     width, color=C_PINK,  label="Female", zorder=3, linewidth=0)

for bars in [bars_t, bars_m, bars_f]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=8.5, color="#444444")

ax.set_xticks(x)
ax.set_xticklabels(edu_labels_short)
ax.set_ylim(0, 13)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
ax.set_ylabel("Unemployment rate (%)")
ax.set_title("The education paradox — unemployment by education & gender (2024)", fontweight="500")
ax.legend(frameon=False, fontsize=10)

# Annotation arrow
ax.annotate("Higher education → higher\nunemployment in Sri Lanka",
            xy=(3, 7.3), xytext=(2.3, 11.5),
            fontsize=9, color="#444444",
            arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart4_education_paradox.png"))
plt.show()
print("Saved → charts_unemployment/chart4_education_paradox.png")


# %% [markdown]
# ## Cell 6 — Chart 5: District unemployment rate (2024)

# %%
dist_2024 = (df_dist[df_dist["Year"] == 2024]
             .drop_duplicates(subset="District")
             .sort_values("Unemployment_Rate", ascending=True))

colors_dist = [C_RED if v >= 6 else C_AMBER if v >= 4 else C_GRAY
               for v in dist_2024["Unemployment_Rate"]]

fig, ax = plt.subplots(figsize=(9, 8))
ax.barh(dist_2024["District"], dist_2024["Unemployment_Rate"],
        color=colors_dist, zorder=3, linewidth=0)

ax.axvline(dist_2024["Unemployment_Rate"].mean(), color="#aaaaaa",
           linewidth=1, linestyle="--")
ax.text(dist_2024["Unemployment_Rate"].mean() + 0.05,
        len(dist_2024) - 1, "National avg", fontsize=8.5, color="#666666")

ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_xlabel("Unemployment rate (%)")
ax.set_title("Unemployment rate by district — 2024", fontweight="500")

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=C_RED,   label="≥ 6%  (high)"),
                   Patch(facecolor=C_AMBER, label="4–6%  (medium)"),
                   Patch(facecolor=C_GRAY,  label="< 4%  (low)")]
ax.legend(handles=legend_elements, frameon=False, fontsize=9, loc="lower right")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart5_district.png"))
plt.show()
print("Saved → charts_unemployment/chart5_district.png")


# %% [markdown]
# ## Cell 7 — Chart 6: Youth vs adult unemployment gap (2015–2024)

# %%
youth = df_age[df_age["Age_Group"] == "15-24"].sort_values("Year")
adult = df_age[df_age["Age_Group"] == "40+"].sort_values("Year")

gap = youth["Total"].values - adult["Total"].values

fig, ax = plt.subplots(figsize=(10, 4.5))

ax.fill_between(youth["Year"], gap, alpha=0.15, color=C_RED)
ax.plot(youth["Year"], gap, marker="o", markersize=5, color=C_RED, linewidth=2, label="Youth–adult gap (pp)")
ax.plot(youth["Year"], youth["Total"], marker="s", markersize=4, color=C_AMBER, linewidth=1.5, linestyle="--", label="Youth 15–24 rate")
ax.plot(adult["Year"], adult["Total"], marker="D", markersize=4, color=C_GRAY,  linewidth=1.5, linestyle="-.", label="Adult 40+ rate")

ax.axvline(2020, color="#aaaaaa", linewidth=0.8, linestyle="--")
ax.text(2020.1, 26, "Covid / crisis", fontsize=9, color="#666666")

ax.set_xticks(YEARS)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
ax.set_ylabel("Rate / gap (percentage points)")
ax.set_title("Youth vs adult unemployment gap — Sri Lanka (2015–2024)", fontweight="500")
ax.legend(frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart6_youth_adult_gap.png"))
plt.show()
print("Saved → charts_unemployment/chart6_youth_adult_gap.png")


# %% [markdown]
# ## Cell 8 — Summary 2×2 panel

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Sri Lanka unemployment analysis — key panels (2015–2024)",
             fontsize=13, fontweight="500", y=1.01)

# A — Age groups
ax = axes[0, 0]
for ag, col, mk in zip(age_groups, colors, markers):
    sub = df_age[df_age["Age_Group"] == ag].sort_values("Year")
    ax.plot(sub["Year"], sub["Total"], marker=mk, markersize=4, color=col, linewidth=2, label=ag)
ax.axvline(2020, color="#aaaaaa", linewidth=0.8, linestyle="--")
ax.set_xticks(YEARS); ax.tick_params(axis="x", labelrotation=45, labelsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_title("A — By age group", fontweight="500", fontsize=10)
ax.legend(frameon=False, fontsize=9)

# B — Education (2024 bar)
ax = axes[0, 1]
x = np.arange(len(edu_order_short))
ax.bar(x - 0.25, edu_2024["Total_Rate"], 0.25, color=C_GRAY,  label="Total",  linewidth=0)
ax.bar(x,         edu_2024["Male"],       0.25, color=C_BLUE,  label="Male",   linewidth=0)
ax.bar(x + 0.25,  edu_2024["Female"],     0.25, color=C_PINK,  label="Female", linewidth=0)
ax.set_xticks(x); ax.set_xticklabels(edu_labels_short, fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_title("B — Education paradox (2024)", fontweight="500", fontsize=10)
ax.legend(frameon=False, fontsize=9)

# C — NEET
ax = axes[1, 0]
ax.plot(neet_female["Year"], neet_female["NEET_Rate"], marker="o", markersize=4, color=C_PINK, linewidth=2, label="Female")
ax.plot(neet_male["Year"],   neet_male["NEET_Rate"],   marker="o", markersize=4, color=C_BLUE, linewidth=2, label="Male")
ax.plot(neet_total["Year"],  neet_total["NEET_Rate"],  marker="D", markersize=3, color=C_GRAY, linewidth=1.5, linestyle="-.", label="Total")
ax.axvline(2020, color="#aaaaaa", linewidth=0.8, linestyle="--")
ax.set_xticks(sorted(neet_total["Year"].unique())); ax.tick_params(axis="x", labelrotation=45, labelsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_title("C — NEET rate by gender", fontweight="500", fontsize=10)
ax.legend(frameon=False, fontsize=9)

# D — District top 10 (2024)
ax = axes[1, 1]
top10 = dist_2024.nlargest(10, "Unemployment_Rate")
c10 = [C_RED if v >= 6 else C_AMBER for v in top10["Unemployment_Rate"]]
ax.barh(top10["District"], top10["Unemployment_Rate"], color=c10, linewidth=0)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.tick_params(axis="y", labelsize=9)
ax.set_title("D — Top 10 districts (2024)", fontweight="500", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "chart8_summary_panel.png"))
plt.show()
print("Saved → charts_unemployment/chart8_summary_panel.png")
