"""
sensitivity_analysis.py
=======================
Deep sensitivity analysis: hours-based vs qualification-based underemployment.
Produces SHAP ranking comparison, sub-period Spearman rho, and qualification trend.

Outputs:
  Visualizations/sensitivity_shap_comparison.png
  Visualizations/sensitivity_qual_trend.png
  output/tables/sensitivity_shap_table.tex
  output/tables/sensitivity_spearman.tex
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os

import xgboost as xgb
import shap
from scipy.stats import spearmanr

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = '/home/kusal/Documents/Reaserch_DS'
VIS_DIR = os.path.join(BASE, 'Visualizations')
TAB_DIR = os.path.join(BASE, 'output', 'tables')
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────
master = pd.read_csv(os.path.join(BASE, 'ardl_vecm', 'master_dataset.csv'))
master.columns = master.columns.str.strip()
for col in master.columns:
    if col != 'Year':
        master[col] = pd.to_numeric(master[col], errors='coerce')
master['Year'] = master['Year'].astype(int)

qual = pd.read_csv(os.path.join(BASE, 'extraction', 'qualification_underemployment.csv'))
qual['Year'] = qual['Year'].astype(int)

df = pd.merge(master, qual, on='Year', how='inner')
df = df[df['Year'].between(2015, 2024)].sort_values('Year').reset_index(drop=True)
print(f"Merged dataset: n={len(df)}, years={df['Year'].min()}-{df['Year'].max()}")

# ── 2. FEATURE SETUP ─────────────────────────────────────────────────────────
FEATURES = [
    'GDP_Growth_Rate',
    'Inflation_Rate',
    'Exchange_Rate_LKR_USD',
    'Youth_LFPR_15_24',
    'Informal_Pct',
    'Remit_Personal_remittances_received_current_US$',
    'AgriProdIdx_Agriculture',
]

# Short labels for display
LABELS = {
    'GDP_Growth_Rate': 'GDP Growth',
    'Inflation_Rate': 'Inflation',
    'Exchange_Rate_LKR_USD': 'Exchange Rate',
    'Youth_LFPR_15_24': 'Youth LFPR',
    'Informal_Pct': 'Informal Emp.',
    'Remit_Personal_remittances_received_current_US$': 'Remittances',
    'AgriProdIdx_Agriculture': 'Agri. Output',
}

# Verify features exist
missing = [f for f in FEATURES if f not in df.columns]
if missing:
    print(f"WARNING: Missing features: {missing}")
    FEATURES = [f for f in FEATURES if f in df.columns]

X = df[FEATURES].copy()
X = X.fillna(X.mean())  # Handle 2024 GDP NaN

y_hours = df['Underemployment_Rate'].astype(float)
y_qual = df['Qual_Underemployment_Rate'].astype(float)

print(f"Features ({len(FEATURES)}): {[LABELS.get(f, f) for f in FEATURES]}")

# ── 3. PARALLEL XGBOOST MODELS ──────────────────────────────────────────────
params = dict(n_estimators=100, max_depth=3, learning_rate=0.1,
              subsample=0.8, random_state=42, verbosity=0)

model_A = xgb.XGBRegressor(**params)
model_A.fit(X, y_hours)

model_B = xgb.XGBRegressor(**params)
model_B.fit(X, y_qual)

print(f"\nModel A (hours-based) R² = {model_A.score(X, y_hours):.4f}")
print(f"Model B (qualification) R² = {model_B.score(X, y_qual):.4f}")

# ── 4. SHAP EXTRACTION & RANKING TABLE ──────────────────────────────────────
exp_A = shap.TreeExplainer(model_A)
exp_B = shap.TreeExplainer(model_B)
sv_A = exp_A.shap_values(X)
sv_B = exp_B.shap_values(X)

imp_A = np.abs(sv_A).mean(axis=0)
imp_B = np.abs(sv_B).mean(axis=0)
imp_A_pct = imp_A / imp_A.sum() * 100
imp_B_pct = imp_B / imp_B.sum() * 100

comp = pd.DataFrame({
    'Feature': [LABELS.get(f, f) for f in FEATURES],
    'Hours |SHAP| %': imp_A_pct,
    'Qual |SHAP| %': imp_B_pct,
})
comp['Rank (Hours)'] = comp['Hours |SHAP| %'].rank(ascending=False).astype(int)
comp['Rank (Qual)'] = comp['Qual |SHAP| %'].rank(ascending=False).astype(int)
comp['|Δ Rank|'] = (comp['Rank (Hours)'] - comp['Rank (Qual)']).abs().astype(int)
comp = comp.sort_values('Rank (Hours)')

print("\n── SHAP Ranking Comparison ──")
print(comp.to_string(index=False))

# Overall Spearman rho
rho_overall, p_overall = spearmanr(comp['Rank (Hours)'], comp['Rank (Qual)'])
print(f"\nOverall Spearman ρ = {rho_overall:.3f} (p = {p_overall:.4f})")

# ── 5. SUB-PERIOD SPEARMAN ──────────────────────────────────────────────────
periods = {
    'Pre-crisis\n(2015–2019)': (2015, 2019),
    'Crisis\n(2020–2022)': (2020, 2022),
    'Post-crisis\n(2023–2024)': (2023, 2024),
}

subperiod_results = []
for label, (y_start, y_end) in periods.items():
    mask = df['Year'].between(y_start, y_end)
    sub = df[mask]
    n_sub = len(sub)

    if n_sub < 3:
        subperiod_results.append({
            'Period': label, 'n': n_sub, 'rho': np.nan, 'p': np.nan,
            'note': 'Insufficient (n<3)'
        })
        continue

    X_sub = sub[FEATURES].apply(pd.to_numeric, errors='coerce').fillna(sub[FEATURES].mean())
    corr_h = X_sub.corrwith(sub['Underemployment_Rate'].astype(float))
    corr_q = X_sub.corrwith(sub['Qual_Underemployment_Rate'].astype(float))

    rank_h = corr_h.abs().rank(ascending=False)
    rank_q = corr_q.abs().rank(ascending=False)

    rho_sp, p_sp = spearmanr(rank_h, rank_q)
    subperiod_results.append({
        'Period': label, 'n': n_sub, 'rho': rho_sp, 'p': p_sp, 'note': ''
    })

sp_df = pd.DataFrame(subperiod_results)
print("\n── Sub-Period Spearman ──")
print(sp_df.to_string(index=False))

# ── 6. QUALIFICATION-BASED TREND ────────────────────────────────────────────
years = df['Year'].values.astype(float)
qual_vals = df['Qual_Underemployment_Rate'].values

coeffs = np.polyfit(years, qual_vals, 1)
trend_line = np.polyval(coeffs, years)
slope = coeffs[0]
print(f"\nQualification-based trend: slope = {slope:.2f} pp/year")
print(f"  2015 fitted: {np.polyval(coeffs, 2015):.1f}%,  2024 fitted: {np.polyval(coeffs, 2024):.1f}%")

# ── 7. VISUALIZATIONS ────────────────────────────────────────────────────────

# --- Figure 1: SHAP Comparison (2-panel) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle('Sensitivity Analysis: Hours-Based vs Qualification-Based Underemployment',
             fontsize=13, fontweight='bold')

# Panel A: Side-by-side bar chart
y_pos = np.arange(len(comp))
width = 0.35
bars1 = ax1.barh(y_pos - width/2, comp['Hours |SHAP| %'], width,
                  color='#2563EB', label='Hours-Based', alpha=0.85)
bars2 = ax1.barh(y_pos + width/2, comp['Qual |SHAP| %'], width,
                  color='#DC2626', label='Qualification-Based', alpha=0.85)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(comp['Feature'], fontsize=9)
ax1.set_xlabel('Relative SHAP Importance (%)', fontsize=10)
ax1.set_title('(A) SHAP Feature Importance by Definition', fontsize=11, fontweight='bold')
ax1.legend(loc='lower right', fontsize=8)
ax1.invert_yaxis()

# Annotate rank deltas
for i, row in comp.reset_index(drop=True).iterrows():
    if row['|Δ Rank|'] >= 2:
        ax1.annotate(f"Δ={int(row['|Δ Rank|'])}",
                     xy=(max(row['Hours |SHAP| %'], row['Qual |SHAP| %']) + 1, i),
                     fontsize=8, color='red', fontweight='bold')

# Panel B: Sub-period Spearman bars
valid_sp = sp_df.dropna(subset=['rho'])
if len(valid_sp) > 0:
    colors = ['#3B82F6', '#EF4444', '#9CA3AF']
    bar_colors = []
    for _, r in sp_df.iterrows():
        if np.isnan(r['rho']):
            bar_colors.append('#D1D5DB')
        elif 'Pre' in r['Period']:
            bar_colors.append('#3B82F6')
        elif 'Crisis' in r['Period']:
            bar_colors.append('#EF4444')
        else:
            bar_colors.append('#9CA3AF')

    rho_vals = sp_df['rho'].fillna(0).values
    period_labels = sp_df['Period'].values
    x_pos = np.arange(len(sp_df))

    ax2.bar(x_pos, rho_vals, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(period_labels, fontsize=9)
    ax2.set_ylabel('Spearman ρ', fontsize=10)
    ax2.set_title('(B) Definition Agreement by Sub-Period', fontsize=11, fontweight='bold')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylim(-1.1, 1.1)
    ax2.axhline(y=rho_overall, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax2.text(len(sp_df)-0.5, rho_overall + 0.05, f'Overall ρ={rho_overall:.2f}',
             fontsize=8, color='green', ha='right')

    # Annotate bars with values
    for i, r in sp_df.iterrows():
        if not np.isnan(r['rho']):
            ax2.text(i, r['rho'] + 0.05, f'{r["rho"]:.2f}',
                     ha='center', fontsize=9, fontweight='bold')
        else:
            ax2.text(i, 0.05, f'n={int(r["n"])}\n(insuff.)',
                     ha='center', fontsize=8, color='gray', style='italic')

plt.tight_layout()
fig.savefig(os.path.join(VIS_DIR, 'sensitivity_shap_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()
print("\nSaved: sensitivity_shap_comparison.png")

# --- Figure 2: Qualification Trend ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df['Year'], qual_vals, 'o-', color='#DC2626', linewidth=2, markersize=8,
        label='Qualification-Based UE Rate', zorder=3)
ax.plot(df['Year'], trend_line, '--', color='#DC2626', alpha=0.5,
        label=f'Linear Trend ({slope:+.2f} pp/yr)')
ax.axhline(y=40, color='#6B7280', linestyle=':', linewidth=1.5,
           label='ILO Global Average (~40%)')
ax.fill_between(df['Year'], 40, qual_vals, alpha=0.1, color='#DC2626')

# Crisis shading
ax.axvspan(2019.5, 2022.5, alpha=0.1, color='red', label='Crisis Period')

ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Qualification Mismatch Rate (%)', fontsize=11)
ax.set_title('Qualification-Based Underemployment: Structural Trend (2015–2024)',
             fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.set_ylim(35, 75)
ax.set_xticks(df['Year'])
ax.grid(alpha=0.3)

# Annotate key values
ax.annotate(f'{qual_vals[0]:.1f}%', (df['Year'].iloc[0], qual_vals[0]),
            textcoords='offset points', xytext=(10, 10), fontsize=9, fontweight='bold')
ax.annotate(f'{qual_vals[-1]:.1f}%', (df['Year'].iloc[-1], qual_vals[-1]),
            textcoords='offset points', xytext=(10, 10), fontsize=9, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(VIS_DIR, 'sensitivity_qual_trend.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: sensitivity_qual_trend.png")

# ── 8. LATEX TABLES ──────────────────────────────────────────────────────────

# SHAP comparison table
with open(os.path.join(TAB_DIR, 'sensitivity_shap_table.tex'), 'w') as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\caption{SHAP Feature Importance: Hours-Based vs Qualification-Based}\n")
    f.write("\\label{tab:shap_sensitivity}\n")
    f.write("\\setlength{\\tabcolsep}{3pt}\n")
    f.write("\\footnotesize\n")
    f.write("\\begin{tabular}{lrrrrc}\n")
    f.write("\\toprule\n")
    f.write("\\textbf{Feature} & \\textbf{Hours \\%} & \\textbf{Rank} & "
            "\\textbf{Qual \\%} & \\textbf{Rank} & \\textbf{$|\\Delta|$}\\\\\n")
    f.write("\\midrule\n")
    for _, row in comp.iterrows():
        marker = '$\\dagger$' if row['|Δ Rank|'] >= 2 else ''
        f.write(f"{row['Feature']}{marker} & {row['Hours |SHAP| %']:.1f} & "
                f"{int(row['Rank (Hours)'])} & {row['Qual |SHAP| %']:.1f} & "
                f"{int(row['Rank (Qual)'])} & {int(row['|Δ Rank|'])}\\\\\n")
    f.write("\\midrule\n")
    f.write(f"\\multicolumn{{6}}{{l}}{{\\scriptsize Spearman $\\rho = {rho_overall:.3f}$ "
            f"($p = {p_overall:.3f}$). $\\dagger$Rank shift $\\geq 2$.}}\\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")
print("Saved: sensitivity_shap_table.tex")

# Sub-period Spearman table
with open(os.path.join(TAB_DIR, 'sensitivity_spearman.tex'), 'w') as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\caption{Sub-Period Definition Agreement (Spearman $\\rho$)}\n")
    f.write("\\label{tab:spearman_subperiod}\n")
    f.write("\\footnotesize\n")
    f.write("\\begin{tabular}{lccc}\n")
    f.write("\\toprule\n")
    f.write("\\textbf{Period} & \\textbf{n} & \\textbf{$\\rho$} & \\textbf{$p$}\\\\\n")
    f.write("\\midrule\n")
    for _, row in sp_df.iterrows():
        period_clean = row['Period'].replace('\n', ' ')
        if np.isnan(row['rho']):
            f.write(f"{period_clean} & {int(row['n'])} & -- & -- \\\\\n")
        else:
            f.write(f"{period_clean} & {int(row['n'])} & {row['rho']:.3f} & {row['p']:.3f}\\\\\n")
    f.write("\\midrule\n")
    f.write(f"Overall & {len(df)} & {rho_overall:.3f} & {p_overall:.3f}\\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")
print("Saved: sensitivity_spearman.tex")

print("\n✓ Sensitivity analysis complete.")
