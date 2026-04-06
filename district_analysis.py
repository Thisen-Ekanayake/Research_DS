"""
district_analysis.py
====================
District-level preliminary analysis using DCS Table 6.4.
Produces: heatmap (25 districts x 3 years), informal→underemployment
regression, and priority district profiles.

Outputs:
  Visualizations/district_heatmap.png
  Visualizations/district_informal_regression.png
  Visualizations/district_profiles.png
  output/tables/district_summary.tex
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')
import os
from scipy.stats import linregress

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = '/home/kusal/Documents/Reaserch_DS'
VIS_DIR = os.path.join(BASE, 'Visualizations')
TAB_DIR = os.path.join(BASE, 'output', 'tables')
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────
dist = pd.read_csv(os.path.join(
    BASE, 'DataLoader',
    'Table_6_4_Underemployment_Unemployment_Informal_Employment_by_District.csv'))

dist.columns = dist.columns.str.strip()
dist['Year'] = pd.to_numeric(dist['Year'], errors='coerce').astype(int)
dist['Underemployment_Rate'] = pd.to_numeric(dist['Underemployment_Rate'], errors='coerce')
dist['Unemployment_Rate'] = pd.to_numeric(dist['Unemployment_Rate'], errors='coerce')
dist['Percentage_of_Informal_Employment'] = pd.to_numeric(
    dist['Percentage_of_Informal_Employment'], errors='coerce')

# Drop aggregate and empty rows
dist = dist[~dist['District'].isin(['All Island', 'Uva'])].copy()
dist['District'] = dist['District'].str.strip()

print(f"District data: {len(dist)} rows, {dist['District'].nunique()} districts, "
      f"years {dist['Year'].min()}-{dist['Year'].max()}")
print(f"Districts: {sorted(dist['District'].unique())}")

# ── 2. HEATMAP (25 districts x 3 key years) ─────────────────────────────────
key_years = [2019, 2022, 2024]
snap = dist[dist['Year'].isin(key_years)].copy()

# Pivot: districts as rows, years as columns
pivot = snap.pivot_table(index='District', columns='Year',
                         values='Underemployment_Rate')

# Sort by 2022 value (descending)
if 2022 in pivot.columns:
    pivot = pivot.sort_values(2022, ascending=True)

fig, ax = plt.subplots(figsize=(8, 12))
data = pivot.values
n_dist, n_years = data.shape

# Custom colormap: white to orange to red
cmap = plt.cm.YlOrRd
norm = mcolors.Normalize(vmin=0, vmax=np.nanmax(data))

im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')

# Annotate cells
for i in range(n_dist):
    for j in range(n_years):
        val = data[i, j]
        if np.isnan(val):
            ax.text(j, i, '—', ha='center', va='center', fontsize=8, color='gray')
        else:
            text_color = 'white' if val > np.nanmax(data) * 0.6 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=8, fontweight='bold', color=text_color)

ax.set_xticks(range(n_years))
ax.set_xticklabels([str(y) for y in pivot.columns], fontsize=11, fontweight='bold')
ax.set_yticks(range(n_dist))
ax.set_yticklabels(pivot.index, fontsize=9)
ax.set_title('District-Level Underemployment Rate (%)\nPre-Crisis (2019) vs Peak (2022) vs Recovery (2024)',
             fontsize=13, fontweight='bold', pad=15)

cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label('Underemployment Rate (%)', fontsize=10)

plt.tight_layout()
fig.savefig(os.path.join(VIS_DIR, 'district_heatmap.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: district_heatmap.png")

# ── 3. INFORMAL → UNDEREMPLOYMENT REGRESSION ────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

colors = {2019: '#3B82F6', 2022: '#EF4444'}
priority_districts = ['Colombo', 'Gampaha', 'Hambantota', 'Jaffna']

reg_results = {}
for year, color in colors.items():
    sub = dist[(dist['Year'] == year)].dropna(
        subset=['Underemployment_Rate', 'Percentage_of_Informal_Employment'])

    x_vals = sub['Percentage_of_Informal_Employment'].values
    y_vals = sub['Underemployment_Rate'].values

    if len(sub) < 3:
        continue

    slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
    reg_results[year] = {
        'slope': slope, 'intercept': intercept,
        'R²': r_value**2, 'p': p_value, 'n': len(sub)
    }

    ax.scatter(x_vals, y_vals, color=color, alpha=0.7, s=50, zorder=3,
               label=f'{year} (R²={r_value**2:.3f}, p={p_value:.3f})')

    # Regression line
    x_line = np.linspace(x_vals.min() - 2, x_vals.max() + 2, 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color=color, linewidth=1.5, linestyle='--', alpha=0.7)

    # Label priority districts
    for _, row in sub.iterrows():
        if row['District'] in priority_districts:
            ax.annotate(row['District'],
                        (row['Percentage_of_Informal_Employment'],
                         row['Underemployment_Rate']),
                        fontsize=8, fontweight='bold', color=color,
                        textcoords='offset points', xytext=(5, 5))

ax.set_xlabel('Informal Employment (%)', fontsize=11)
ax.set_ylabel('Underemployment Rate (%)', fontsize=11)
ax.set_title('District-Level: Informal Employment vs Underemployment\n'
             'Pre-Crisis (2019, blue) vs Crisis Peak (2022, red)',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(VIS_DIR, 'district_informal_regression.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: district_informal_regression.png")

print("\n── Regression Results ──")
for year, res in reg_results.items():
    print(f"  {year}: slope={res['slope']:.4f}, R²={res['R²']:.3f}, "
          f"p={res['p']:.4f}, n={res['n']}")

# ── 4. PRIORITY DISTRICT PROFILES ───────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
fig.suptitle('Priority District Underemployment Trajectories (2015–2024)',
             fontsize=13, fontweight='bold')

for ax, district in zip(axes.flat, priority_districts):
    d = dist[dist['District'] == district].sort_values('Year')
    if len(d) == 0:
        ax.set_title(f'{district} (no data)')
        continue

    ax.plot(d['Year'], d['Underemployment_Rate'], 'o-', color='#2563EB',
            linewidth=2, markersize=6, zorder=3)
    ax.axvspan(2019.5, 2022.5, alpha=0.1, color='red', label='Crisis')
    ax.set_title(district, fontsize=11, fontweight='bold')
    ax.set_ylabel('UE Rate (%)', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(range(2015, 2025))
    ax.tick_params(axis='x', rotation=45)

    # Annotate informal employment for context
    inf = dist[(dist['District'] == district) & (dist['Year'] == 2022)]
    if len(inf) > 0:
        inf_val = inf['Percentage_of_Informal_Employment'].iloc[0]
        if not np.isnan(inf_val):
            ax.text(0.95, 0.95, f'Informal: {inf_val:.0f}%',
                    transform=ax.transAxes, fontsize=8, va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
fig.savefig(os.path.join(VIS_DIR, 'district_profiles.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: district_profiles.png")

# ── 5. LATEX TABLE ───────────────────────────────────────────────────────────
with open(os.path.join(TAB_DIR, 'district_summary.tex'), 'w') as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\caption{District-Level Informal Employment and Underemployment Regression}\n")
    f.write("\\label{tab:district_regression}\n")
    f.write("\\footnotesize\n")
    f.write("\\begin{tabular}{lcccc}\n")
    f.write("\\toprule\n")
    f.write("\\textbf{Year} & \\textbf{$n$} & \\textbf{Slope} & "
            "\\textbf{$R^2$} & \\textbf{$p$}\\\\\n")
    f.write("\\midrule\n")
    for year, res in sorted(reg_results.items()):
        f.write(f"{year} & {res['n']} & {res['slope']:.4f} & "
                f"{res['R²']:.3f} & {res['p']:.4f}\\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")
print("Saved: district_summary.tex")

print("\n✓ District analysis complete.")
