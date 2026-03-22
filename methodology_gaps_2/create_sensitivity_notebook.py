"""
create_sensitivity_notebook.py
==============================
Generates Data_Analysis/Sensitivity_Analysis_Underemployment_Definitions.ipynb
with a complete, executable sensitivity analysis comparing:
  1. Time-related underemployment (TRU, hours-based, ILO/DCS)
  2. Qualification-based underemployment (occupational mismatch proxy, DCS LFS)
"""

import nbformat as nbf
import os

OUT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'Data_Analysis',
    'Sensitivity_Analysis_Underemployment_Definitions.ipynb'
)
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

nb = nbf.v4.new_notebook()

# ─────────────────────────────────────────────────────────────────────────────
cells = []

cells.append(nbf.v4.new_markdown_cell("""\
# Sensitivity Analysis: Underemployment Definitions

**Project:** Analysing the Economic Drivers of Underemployment in Sri Lanka (2015–2024)

This notebook delivers the **sensitivity analysis** promised in the research proposal:
> *A sensitivity analysis comparing findings across both underemployment definitions.*

### Definitions compared
| # | Definition | Measure | Source |
|---|---|---|---|
| 1 | **Time-related underemployment (TRU)** | Employed persons working <40 hrs/week who desire more hours (%) | ILO/DCS LFS annual |
| 2 | **Qualification-based underemployment** | Employed persons in occupations below their educational level (%) | DCS LFS occupational mismatch proxy |

### Research questions answered here
- Do GDP contraction and inflation drive *both* definitions, or only the hours-based series?
- Does the 2022 crisis break appear in the qualification-based series as well?
- Does SHAP feature ranking converge or diverge across definitions?
"""))

cells.append(nbf.v4.new_markdown_cell("## 1. Setup"))

cells.append(nbf.v4.new_code_cell("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('deep')

MASTER_PATH = '../ardl_vecm/master_dataset.csv'
QUAL_PATH   = '../extraction/qualification_underemployment.csv'
OUT_DIR     = '../Visualizations'

import os
os.makedirs(OUT_DIR, exist_ok=True)

master = pd.read_csv(MASTER_PATH)
master.columns = [c.strip().lower() for c in master.columns]
master['year']  = master['year'].astype(int)

qual = pd.read_csv(QUAL_PATH)
qual.columns = [c.strip().lower() for c in qual.columns]
qual['year'] = qual['year'].astype(int)

df = pd.merge(master, qual[['year','qual_underemployment_rate',
                              'qual_underemployment_male',
                              'qual_underemployment_female']],
              on='year', how='inner')
df = df[df['year'].between(2015, 2024)].sort_values('year').reset_index(drop=True)

print(f"Merged dataset: {df.shape[0]} rows, {df.shape[1]} cols")
print(f"Years: {df['year'].min()}–{df['year'].max()}")

# Resolve target and predictor column names robustly
TRU_COL   = next((c for c in df.columns if 'tru_total' in c or
                  (c.startswith('tru') and 'female' not in c and 'male' not in c)),
                 None)
if TRU_COL is None:
    # Fallback: use ILO female+male mean
    if 'tru_female' in df.columns and 'tru_male' in df.columns:
        df['tru_total'] = (df['tru_female'] + df['tru_male']) / 2
        TRU_COL = 'tru_total'
    else:
        TRU_COL = 'underemployment_total'

QUAL_COL  = 'qual_underemployment_rate'

PREDICTORS = [c for c in [
    'gdp_growth_pct', 'inflation_cpi_pct', 'exchange_rate_lkr_usd',
    'youth_lfpr_pct', 'informal_emp_pct',
    'remittances_usd', 'agri_output_index', 'parttime_emp_pct',
] if c in df.columns]

print(f"TRU target:  {TRU_COL}")
print(f"Qual target: {QUAL_COL}")
print(f"Predictors:  {PREDICTORS}")
df[[TRU_COL, QUAL_COL] + PREDICTORS[:3]].head()
"""))

cells.append(nbf.v4.new_markdown_cell("## 2. Comparative EDA — divergent trajectories"))

cells.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: both series on dual axis
ax1 = axes[0]
ax2 = ax1.twinx()

ln1 = ax1.plot(df['year'], df[TRU_COL], 'o-', color='#2563EB', lw=2, label='TRU (hours-based, left)')
ln2 = ax2.plot(df['year'], df[QUAL_COL], 's--', color='#DC2626', lw=2, label='Qual-based (right)')
ax1.axvspan(2021.5, 2022.5, alpha=0.12, color='grey', label='2022 crisis')

ax1.set_xlabel('Year'); ax1.set_ylabel('TRU rate (%)', color='#2563EB')
ax2.set_ylabel('Qualification underemployment (%)', color='#DC2626')
ax1.tick_params(axis='y', labelcolor='#2563EB')
ax2.tick_params(axis='y', labelcolor='#DC2626')
lns = ln1 + ln2
ax1.legend(lns, [l.get_label() for l in lns], loc='upper left', fontsize=9)
ax1.set_title('Divergent scales: TRU vs qualification-based', fontweight='bold')

# Right: normalised (z-score) overlay
ax = axes[1]
for col, label, color in [(TRU_COL, 'TRU (z)', '#2563EB'), (QUAL_COL, 'Qual-based (z)', '#DC2626')]:
    z = (df[col] - df[col].mean()) / df[col].std()
    ax.plot(df['year'], z, marker='o', lw=2, label=label, color=color)
ax.axhline(0, color='grey', lw=0.8, ls='--')
ax.axvspan(2021.5, 2022.5, alpha=0.12, color='grey')
ax.set_xlabel('Year'); ax.set_ylabel('Standardised value (z-score)')
ax.set_title('Normalised comparison — co-movement?', fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Sensitivity_EDA_Comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: Sensitivity_EDA_Comparison.png")

# Crisis-phase descriptive stats
df['phase'] = df['year'].apply(lambda y: 'pre-crisis' if y<=2019 else 'crisis' if y<=2022 else 'recovery')
print("\\nDescriptive stats by phase:")
print(df.groupby('phase')[[TRU_COL, QUAL_COL]].agg(['mean','std']).round(2))
"""))

cells.append(nbf.v4.new_markdown_cell("## 3. Correlation comparison — do the same predictors drive both?"))

cells.append(nbf.v4.new_code_cell("""\
corr_tru  = df[PREDICTORS + [TRU_COL]].corr()[TRU_COL].drop(TRU_COL)
corr_qual = df[PREDICTORS + [QUAL_COL]].corr()[QUAL_COL].drop(QUAL_COL)

corr_df = pd.DataFrame({'TRU (hours-based)': corr_tru,
                         'Qualification-based': corr_qual}).round(3)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap
sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=axes[0],
            linewidths=0.5, annot_kws={'size': 11})
axes[0].set_title('Pearson correlations with each definition', fontweight='bold')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=20, ha='right')

# Side-by-side bar
x = np.arange(len(corr_df))
w = 0.35
axes[1].bar(x - w/2, corr_df['TRU (hours-based)'], w,
            color='#2563EB', alpha=0.8, label='TRU')
axes[1].bar(x + w/2, corr_df['Qualification-based'], w,
            color='#DC2626', alpha=0.8, label='Qual-based')
axes[1].axhline(0, color='black', lw=0.8)
axes[1].set_xticks(x); axes[1].set_xticklabels(corr_df.index, rotation=35, ha='right', fontsize=9)
axes[1].set_ylabel('Pearson r'); axes[1].set_title('Correlation magnitude comparison', fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Sensitivity_Correlation_Comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print(corr_df.to_string())
"""))

cells.append(nbf.v4.new_markdown_cell("## 4. Parallel XGBoost + SHAP — do driver rankings converge?"))

cells.append(nbf.v4.new_code_cell("""\
X = df[PREDICTORS].fillna(df[PREDICTORS].mean())
y_tru  = df[TRU_COL]
y_qual = df[QUAL_COL]

# Fit identical architectures
def fit_model(X, y):
    m = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                          subsample=0.8, random_state=42, verbosity=0)
    m.fit(X, y)
    return m

m_tru  = fit_model(X, y_tru)
m_qual = fit_model(X, y_qual)

exp_tru  = shap.TreeExplainer(m_tru)
exp_qual = shap.TreeExplainer(m_qual)
sv_tru   = exp_tru.shap_values(X)
sv_qual  = exp_qual.shap_values(X)

imp_tru  = np.abs(sv_tru).mean(axis=0)
imp_qual = np.abs(sv_qual).mean(axis=0)

# Normalise to % of total for cross-model comparison
imp_tru_pct  = imp_tru  / imp_tru.sum()  * 100
imp_qual_pct = imp_qual / imp_qual.sum() * 100

shap_comp = pd.DataFrame({
    'feature':      PREDICTORS,
    'TRU_pct':      imp_tru_pct,
    'Qual_pct':     imp_qual_pct,
}).sort_values('TRU_pct', ascending=False)
shap_comp['rank_tru']  = shap_comp['TRU_pct'].rank(ascending=False).astype(int)
shap_comp['rank_qual'] = shap_comp['Qual_pct'].rank(ascending=False).astype(int)
shap_comp['rank_delta'] = (shap_comp['rank_tru'] - shap_comp['rank_qual']).abs()

print("SHAP importance comparison (% of total):")
print(shap_comp.to_string(index=False))

from scipy.stats import spearmanr
rho, p = spearmanr(shap_comp['rank_tru'], shap_comp['rank_qual'])
print(f"\\nSpearman rank correlation of SHAP rankings: ρ = {rho:.3f}, p = {p:.3f}")
print(f"→ {'Rankings CONVERGE (consistent drivers)' if rho > 0.6 else 'Rankings DIVERGE (definition-sensitive drivers)'}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# Parallel bar plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)

srt = shap_comp.sort_values('TRU_pct')
y_pos = np.arange(len(srt))

axes[0].barh(y_pos, srt['TRU_pct'], color='#2563EB', alpha=0.85)
axes[0].set_yticks(y_pos); axes[0].set_yticklabels(srt['feature'], fontsize=9)
axes[0].set_xlabel('Relative importance (%)')
axes[0].set_title('TRU (hours-based) — SHAP drivers', fontweight='bold')

srt2 = shap_comp.sort_values('Qual_pct')
y_pos2 = np.arange(len(srt2))
axes[1].barh(y_pos2, srt2['Qual_pct'], color='#DC2626', alpha=0.85)
axes[1].set_yticks(y_pos2); axes[1].set_yticklabels(srt2['feature'], fontsize=9)
axes[1].set_xlabel('Relative importance (%)')
axes[1].set_title('Qualification-based — SHAP drivers', fontweight='bold')

fig.suptitle(
    f'SHAP Feature Importance: Sensitivity to Underemployment Definition\\n'
    f'Spearman rank ρ = {rho:.2f} ({"converge" if rho>0.6 else "diverge"})',
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Sensitivity_SHAP_Comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: Sensitivity_SHAP_Comparison.png")
"""))

cells.append(nbf.v4.new_markdown_cell("## 5. Gender gap — does the crisis widen both gaps equally?"))

cells.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# TRU gender gap
if 'tru_female' in df.columns and 'tru_male' in df.columns:
    df['tru_gap'] = df['tru_female'] - df['tru_male']
    axes[0].bar(df['year'], df['tru_gap'], color='#7C3AED', alpha=0.8)
    axes[0].axhline(0, color='black', lw=0.8)
    axes[0].axvspan(2021.5, 2022.5, alpha=0.12, color='grey')
    axes[0].set_title('TRU gender gap (female − male)', fontweight='bold')
    axes[0].set_xlabel('Year'); axes[0].set_ylabel('pp difference')

# Qualification gender gap
if 'qual_underemployment_female' in df.columns and 'qual_underemployment_male' in df.columns:
    df['qual_gap'] = df['qual_underemployment_female'] - df['qual_underemployment_male']
    axes[1].bar(df['year'], df['qual_gap'], color='#D97706', alpha=0.8)
    axes[1].axhline(0, color='black', lw=0.8)
    axes[1].axvspan(2021.5, 2022.5, alpha=0.12, color='grey')
    axes[1].set_title('Qualification-based gender gap (female − male)', fontweight='bold')
    axes[1].set_xlabel('Year'); axes[1].set_ylabel('pp difference')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/Sensitivity_Gender_Gap_Comparison.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 6. Summary of sensitivity findings"))

cells.append(nbf.v4.new_code_cell("""\
print("=" * 65)
print("SENSITIVITY ANALYSIS SUMMARY")
print("=" * 65)

tru_mean  = df[TRU_COL].mean()
qual_mean = df[QUAL_COL].mean()
tru_crisis  = df[df['phase']=='crisis'][TRU_COL].mean()
qual_crisis = df[df['phase']=='crisis'][QUAL_COL].mean()
tru_pre     = df[df['phase']=='pre-crisis'][TRU_COL].mean()
qual_pre    = df[df['phase']=='pre-crisis'][QUAL_COL].mean()

top_tru  = shap_comp.sort_values('TRU_pct', ascending=False).iloc[0]['feature']
top_qual = shap_comp.sort_values('Qual_pct', ascending=False).iloc[0]['feature']

print(f"\\n1. SCALE DIFFERENCE")
print(f"   TRU mean: {tru_mean:.2f}%  |  Qual mean: {qual_mean:.2f}%")
print(f"   TRU is ~{qual_mean/tru_mean:.0f}x smaller in absolute magnitude.")

print(f"\\n2. CRISIS SENSITIVITY")
print(f"   TRU:  pre-crisis {tru_pre:.2f}% → crisis {tru_crisis:.2f}% "
      f"(+{tru_crisis-tru_pre:.2f} pp, {(tru_crisis/tru_pre-1)*100:.0f}% rise)")
print(f"   Qual: pre-crisis {qual_pre:.2f}% → crisis {qual_crisis:.2f}% "
      f"(+{qual_crisis-qual_pre:.2f} pp, {(qual_crisis/qual_pre-1)*100:.0f}% rise)")
print(f"   TRU is {'more' if (tru_crisis/tru_pre) > (qual_crisis/qual_pre) else 'less'} crisis-sensitive in % terms.")

print(f"\\n3. DRIVER CONVERGENCE")
print(f"   Top TRU driver:  {top_tru}")
print(f"   Top Qual driver: {top_qual}")
print(f"   Spearman ρ = {rho:.3f} → "
      f"{'CONVERGENT: same macro drivers dominate both' if rho>0.6 else 'DIVERGENT: definition matters for policy'}")

print(f"\\n4. POLICY IMPLICATION")
if rho > 0.6:
    print("   GDP contraction and inflation as dominant drivers is ROBUST to")
    print("   definition choice — policy targeting these variables protects")
    print("   against both forms of underemployment.")
else:
    print("   Driver rankings differ across definitions — separate policy")
    print("   instruments are needed for hours-based vs qualification mismatch.")
"""))

nb['cells'] = cells

with open(OUT_FILE, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Generated: {OUT_FILE}")
