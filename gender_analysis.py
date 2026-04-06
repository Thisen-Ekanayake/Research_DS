"""
gender_analysis.py
==================
Gender disaggregation: SHAP comparison (aggregate/female/male), statistical
testing of the gender gap, ZA structural break on the gap series, and
qualification-based gender gap analysis.

Outputs:
  Visualizations/gender_shap_comparison.png
  Visualizations/gender_gap_extended.png
  Visualizations/gender_gap_statistical.png
  output/tables/gender_shap_table.tex
  output/tables/gender_gap_tests.tex
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
from scipy.stats import wilcoxon, shapiro, spearmanr

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = '/home/kusal/Documents/Reaserch_DS'
VIS_DIR = os.path.join(BASE, 'Visualizations')
TAB_DIR = os.path.join(BASE, 'output', 'tables')
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────
# Macro features from master dataset
master = pd.read_csv(os.path.join(BASE, 'ardl_vecm', 'master_dataset.csv'))
master.columns = master.columns.str.strip()
for col in master.columns:
    if col != 'Year':
        master[col] = pd.to_numeric(master[col], errors='coerce')
master['Year'] = master['Year'].astype(int)

# Gender-disaggregated hours-based UE from DCS (authoritative source)
gender_dcs = pd.read_csv(os.path.join(
    BASE, 'labour', 'finalized_csv', 'sl_labour_csv', 'under_em_by_gender.csv'))
gender_dcs.columns = gender_dcs.columns.str.strip()

# Qualification-based UE with gender
qual = pd.read_csv(os.path.join(BASE, 'extraction', 'qualification_underemployment.csv'))
qual['Year'] = qual['Year'].astype(int)

# Merge: use DCS gender values as targets, master for features
df = master[['Year']].copy()
df = df.merge(gender_dcs.rename(columns={
    'year': 'Year', 'male': 'UE_Male', 'female': 'UE_Female', 'total': 'UE_Total'
}), on='Year', how='inner')
df = df.merge(qual[['Year', 'Qual_Underemployment_Rate',
                     'Qual_Underemployment_Male', 'Qual_Underemployment_Female']],
              on='Year', how='inner')

# Add macro features from master
for feat in ['GDP_Growth_Rate', 'Inflation_Rate', 'Exchange_Rate_LKR_USD',
             'Youth_LFPR_15_24', 'Informal_Pct',
             'Remit_Personal_remittances_received_current_US$',
             'AgriProdIdx_Agriculture']:
    if feat in master.columns:
        df[feat] = master.set_index('Year').loc[df['Year'].values, feat].values

df = df.sort_values('Year').reset_index(drop=True)
print(f"Gender dataset: n={len(df)}, years={df['Year'].min()}-{df['Year'].max()}")
print(df[['Year', 'UE_Male', 'UE_Female', 'UE_Total']].to_string(index=False))

# ── 2. FEATURE SETUP ─────────────────────────────────────────────────────────
FEATURES = [
    'GDP_Growth_Rate', 'Inflation_Rate', 'Exchange_Rate_LKR_USD',
    'Youth_LFPR_15_24', 'Informal_Pct',
    'Remit_Personal_remittances_received_current_US$',
    'AgriProdIdx_Agriculture',
]
FEATURES = [f for f in FEATURES if f in df.columns]

LABELS = {
    'GDP_Growth_Rate': 'GDP Growth',
    'Inflation_Rate': 'Inflation',
    'Exchange_Rate_LKR_USD': 'Exchange Rate',
    'Youth_LFPR_15_24': 'Youth LFPR',
    'Informal_Pct': 'Informal Emp.',
    'Remit_Personal_remittances_received_current_US$': 'Remittances',
    'AgriProdIdx_Agriculture': 'Agri. Output',
}

X = df[FEATURES].copy().fillna(df[FEATURES].mean())

# ── 3. GENDER-DISAGGREGATED XGBOOST-SHAP ────────────────────────────────────
params = dict(n_estimators=100, max_depth=3, learning_rate=0.1,
              subsample=0.8, random_state=42, verbosity=0)

targets = {
    'Aggregate': 'UE_Total',
    'Female': 'UE_Female',
    'Male': 'UE_Male',
}

shap_importance = {}
shap_values_all = {}

for name, col in targets.items():
    y = df[col].astype(float)
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)

    exp = shap.TreeExplainer(model)
    sv = exp.shap_values(X)
    imp = np.abs(sv).mean(axis=0)
    shap_importance[name] = imp / imp.sum() * 100
    shap_values_all[name] = sv
    print(f"  Model ({name}): R² = {model.score(X, y):.4f}")

# Build comparison
gender_comp = pd.DataFrame({'Feature': [LABELS.get(f, f) for f in FEATURES]})
for name in targets:
    gender_comp[f'{name} %'] = shap_importance[name]
    gender_comp[f'{name} Rank'] = pd.Series(shap_importance[name]).rank(ascending=False).astype(int).values

gender_comp = gender_comp.sort_values('Aggregate Rank')
print("\n── Gender-Disaggregated SHAP Rankings ──")
print(gender_comp.to_string(index=False))

# Spearman between female and male rankings
rho_fm, p_fm = spearmanr(gender_comp['Female Rank'], gender_comp['Male Rank'])
print(f"\nFemale vs Male SHAP rank Spearman ρ = {rho_fm:.3f} (p = {p_fm:.4f})")

# ── 4. STATISTICAL TEST OF GENDER GAP ───────────────────────────────────────
gap = df['UE_Female'].values - df['UE_Male'].values
mean_gap = np.mean(gap)
std_gap = np.std(gap, ddof=1)

print(f"\n── Gender Gap Statistics ──")
print(f"Mean gap (F-M): {mean_gap:.2f} pp, SD: {std_gap:.2f}")
print(f"Gap by year: {dict(zip(df['Year'], np.round(gap, 1)))}")

# Shapiro-Wilk normality test
sw_stat, sw_p = shapiro(gap)
print(f"Shapiro-Wilk: W={sw_stat:.3f}, p={sw_p:.4f} {'(normal)' if sw_p > 0.05 else '(non-normal)'}")

# Paired Wilcoxon signed-rank test (H0: median gap = 0)
try:
    w_stat, w_p = wilcoxon(gap, alternative='greater')  # one-sided: female > male
    print(f"Wilcoxon signed-rank (one-sided F>M): W={w_stat:.3f}, p={w_p:.4f}")
except ValueError as e:
    w_stat, w_p = np.nan, np.nan
    print(f"Wilcoxon: {e}")

# Bootstrap CI for crisis-peak gap (2020-2022)
crisis_mask = df['Year'].between(2020, 2022)
crisis_gap = gap[crisis_mask]
print(f"\nCrisis-period gap (2020-2022): {crisis_gap}")

np.random.seed(42)
n_boot = 10000
boot_means = np.array([
    np.mean(np.random.choice(crisis_gap, size=len(crisis_gap), replace=True))
    for _ in range(n_boot)
])
ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
print(f"Bootstrap 95% CI for crisis gap: [{ci_lower:.2f}, {ci_upper:.2f}] pp")

# ── 5. ZIVOT-ANDREWS ON GENDER GAP ──────────────────────────────────────────
def zivot_andrews(series_vals, series_index, name):
    """ZA Model C test (adapted from Zivot-Andrews/structural_breaks.py)."""
    y = series_vals.copy()
    n = len(y)
    results = []
    trim = max(int(0.15 * n), 2)

    for tb in range(trim, n - trim):
        du = np.array([1 if t >= tb else 0 for t in range(n)])
        dt = np.array([t - tb if t >= tb else 0 for t in range(n)])
        dy = np.diff(y)
        y_lag = y[:-1]
        du_t = du[1:]
        dt_t = dt[1:]
        X_reg = np.column_stack([np.ones(len(dy)), y_lag, du_t, dt_t])
        try:
            coef, _, _, _ = np.linalg.lstsq(X_reg, dy, rcond=None)
            resid = dy - X_reg @ coef
            sigma2 = np.sum(resid**2) / (len(dy) - X_reg.shape[1])
            cov = sigma2 * np.linalg.inv(X_reg.T @ X_reg)
            t_stat = coef[1] / np.sqrt(cov[1, 1])
            results.append((tb, t_stat))
        except Exception:
            pass

    if not results:
        return {'Series': name, 'Break Year': None, 'ZA t-stat': None,
                'Significance': 'Infeasible'}

    results.sort(key=lambda x: x[1])
    tb_idx, min_t = results[0]
    break_year = series_index[tb_idx]

    cv = {0.01: -5.57, 0.05: -5.08, 0.10: -4.82}
    sig = ("***" if min_t < cv[0.01] else
           ("**" if min_t < cv[0.05] else
            ("*" if min_t < cv[0.10] else "--")))

    return {
        'Series': name, 'Break Year': break_year,
        'ZA t-stat': round(min_t, 4), 'Significance': sig
    }

za_result = zivot_andrews(gap, df['Year'].values, 'Gender Gap (F-M)')
print(f"\n── ZA Structural Break on Gender Gap ──")
print(f"  Break year: {za_result['Break Year']}")
print(f"  ZA t-stat: {za_result['ZA t-stat']}")
print(f"  Significance: {za_result['Significance']}")
print(f"  (CVs: 1%=-5.57, 5%=-5.08, 10%=-4.82; n={len(gap)} -> low power)")

# ── 6. QUALIFICATION-BASED GENDER GAP ───────────────────────────────────────
qual_gap = df['Qual_Underemployment_Female'].values - df['Qual_Underemployment_Male'].values
print(f"\n── Qualification-Based Gender Gap ──")
for yr, g in zip(df['Year'], qual_gap):
    print(f"  {yr}: {g:.2f} pp")
print(f"Range: {qual_gap.min():.2f} to {qual_gap.max():.2f} pp")
print(f"2015 -> 2021 widening: {qual_gap[0]:.2f} -> {qual_gap[df['Year'].values.tolist().index(2021)]:.2f} pp")

# ── 7. VISUALIZATIONS ────────────────────────────────────────────────────────

COLOR_MALE = '#2563EB'
COLOR_FEMALE = '#DC2626'
COLOR_AGG = '#6B7280'

# --- Figure 1: Gender SHAP Comparison ---
fig, ax = plt.subplots(figsize=(12, 6))
n_feat = len(gender_comp)
y_pos = np.arange(n_feat)
width = 0.25

bars1 = ax.barh(y_pos - width, gender_comp['Aggregate %'], width,
                color=COLOR_AGG, label='Aggregate', alpha=0.85)
bars2 = ax.barh(y_pos, gender_comp['Female %'], width,
                color=COLOR_FEMALE, label='Female', alpha=0.85)
bars3 = ax.barh(y_pos + width, gender_comp['Male %'], width,
                color=COLOR_MALE, label='Male', alpha=0.85)

ax.set_yticks(y_pos)
ax.set_yticklabels(gender_comp['Feature'], fontsize=10)
ax.set_xlabel('Relative SHAP Importance (%)', fontsize=11)
ax.set_title('Gender-Disaggregated SHAP Feature Importance\n'
             '(Aggregate vs Female vs Male Underemployment)',
             fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(VIS_DIR, 'gender_shap_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()
print("\nSaved: gender_shap_comparison.png")

# --- Figure 2: Extended Gender Gap (hours + qualification) ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
fig.suptitle('Gender Gap in Underemployment: Hours-Based and Qualification-Based',
             fontsize=13, fontweight='bold')

# Panel A: Hours-based
ax1.plot(df['Year'], df['UE_Female'], 'o-', color=COLOR_FEMALE, linewidth=2,
         markersize=7, label='Female', zorder=3)
ax1.plot(df['Year'], df['UE_Male'], 's-', color=COLOR_MALE, linewidth=2,
         markersize=7, label='Male', zorder=3)
ax1.fill_between(df['Year'], df['UE_Male'], df['UE_Female'],
                 alpha=0.15, color=COLOR_FEMALE, label='Gender Gap')
ax1.axvspan(2019.5, 2022.5, alpha=0.08, color='red')
ax1.set_ylabel('Underemployment Rate (%)', fontsize=10)
ax1.set_title('(A) Hours-Based (Time-Related) Underemployment', fontsize=11)
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(alpha=0.3)

# Annotate peak gap
peak_idx = np.argmax(gap)
ax1.annotate(f'Gap: {gap[peak_idx]:.1f} pp',
             xy=(df['Year'].iloc[peak_idx],
                 (df['UE_Female'].iloc[peak_idx] + df['UE_Male'].iloc[peak_idx]) / 2),
             fontsize=9, fontweight='bold', color=COLOR_FEMALE,
             textcoords='offset points', xytext=(15, 0),
             arrowprops=dict(arrowstyle='->', color=COLOR_FEMALE, lw=1.5))

# Panel B: Qualification-based
ax2.plot(df['Year'], df['Qual_Underemployment_Female'], 'o-', color=COLOR_FEMALE,
         linewidth=2, markersize=7, label='Female', zorder=3)
ax2.plot(df['Year'], df['Qual_Underemployment_Male'], 's-', color=COLOR_MALE,
         linewidth=2, markersize=7, label='Male', zorder=3)
ax2.fill_between(df['Year'], df['Qual_Underemployment_Male'],
                 df['Qual_Underemployment_Female'],
                 alpha=0.15, color=COLOR_FEMALE, label='Gender Gap')
ax2.axvspan(2019.5, 2022.5, alpha=0.08, color='red')
ax2.set_xlabel('Year', fontsize=10)
ax2.set_ylabel('Qualification Mismatch Rate (%)', fontsize=10)
ax2.set_title('(B) Qualification-Based Underemployment', fontsize=11)
ax2.legend(loc='upper left', fontsize=9)
ax2.set_xticks(df['Year'])
ax2.grid(alpha=0.3)

# Annotate peak qualification gap
qual_peak_idx = np.argmax(qual_gap)
ax2.annotate(f'Gap: {qual_gap[qual_peak_idx]:.1f} pp',
             xy=(df['Year'].iloc[qual_peak_idx],
                 (df['Qual_Underemployment_Female'].iloc[qual_peak_idx] +
                  df['Qual_Underemployment_Male'].iloc[qual_peak_idx]) / 2),
             fontsize=9, fontweight='bold', color=COLOR_FEMALE,
             textcoords='offset points', xytext=(15, 0),
             arrowprops=dict(arrowstyle='->', color=COLOR_FEMALE, lw=1.5))

plt.tight_layout()
fig.savefig(os.path.join(VIS_DIR, 'gender_gap_extended.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: gender_gap_extended.png")

# --- Figure 3: Bootstrap distribution ---
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(boot_means, bins=50, color=COLOR_FEMALE, alpha=0.6, edgecolor='white')
ax.axvline(np.mean(crisis_gap), color='black', linewidth=2, linestyle='-',
           label=f'Observed Mean = {np.mean(crisis_gap):.2f} pp')
ax.axvline(ci_lower, color='red', linewidth=1.5, linestyle='--',
           label=f'95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]')
ax.axvline(ci_upper, color='red', linewidth=1.5, linestyle='--')
ax.axvline(0, color='gray', linewidth=1, linestyle=':')
ax.set_xlabel('Gender Gap (Female − Male, pp)', fontsize=10)
ax.set_ylabel('Bootstrap Count', fontsize=10)
ax.set_title('Bootstrap Distribution: Crisis-Period Gender Gap (2020–2022)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(VIS_DIR, 'gender_gap_statistical.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: gender_gap_statistical.png")

# ── 8. LATEX TABLES ──────────────────────────────────────────────────────────

# Gender SHAP comparison
with open(os.path.join(TAB_DIR, 'gender_shap_table.tex'), 'w') as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\caption{Gender-Disaggregated SHAP Feature Rankings}\n")
    f.write("\\label{tab:gender_shap}\n")
    f.write("\\setlength{\\tabcolsep}{3pt}\n")
    f.write("\\footnotesize\n")
    f.write("\\begin{tabular}{lrrrrrr}\n")
    f.write("\\toprule\n")
    f.write("\\textbf{Feature} & \\textbf{Agg \\%} & \\textbf{Rank} & "
            "\\textbf{Female \\%} & \\textbf{Rank} & "
            "\\textbf{Male \\%} & \\textbf{Rank}\\\\\n")
    f.write("\\midrule\n")
    for _, row in gender_comp.iterrows():
        f.write(f"{row['Feature']} & {row['Aggregate %']:.1f} & {int(row['Aggregate Rank'])} & "
                f"{row['Female %']:.1f} & {int(row['Female Rank'])} & "
                f"{row['Male %']:.1f} & {int(row['Male Rank'])}\\\\\n")
    f.write("\\midrule\n")
    f.write(f"\\multicolumn{{7}}{{l}}{{\\scriptsize Female--Male rank $\\rho = {rho_fm:.3f}$ "
            f"($p = {p_fm:.3f}$)}}\\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")
print("Saved: gender_shap_table.tex")

# Gender gap test results
with open(os.path.join(TAB_DIR, 'gender_gap_tests.tex'), 'w') as f:
    f.write("\\begin{table}[ht]\n")
    f.write("\\caption{Statistical Tests on the Gender Gap in Underemployment}\n")
    f.write("\\label{tab:gender_gap_tests}\n")
    f.write("\\footnotesize\n")
    f.write("\\begin{tabular}{llrl}\n")
    f.write("\\toprule\n")
    f.write("\\textbf{Test} & \\textbf{Statistic} & \\textbf{$p$} & \\textbf{Result}\\\\\n")
    f.write("\\midrule\n")
    f.write(f"Mean gap (F$-$M) & {mean_gap:.2f}~pp & -- & "
            f"Persistent female excess\\\\\n")
    f.write(f"Shapiro-Wilk (gap) & $W={sw_stat:.3f}$ & {sw_p:.3f} & "
            f"{'Normal' if sw_p > 0.05 else 'Non-normal'}\\\\\n")
    if not np.isnan(w_stat):
        sig_w = '\\checkmark' if w_p < 0.05 else '--'
        f.write(f"Wilcoxon signed-rank & $W={w_stat:.1f}$ & {w_p:.4f} & "
                f"{'Significant' if w_p < 0.05 else 'Not sig.'}\\\\\n")
    f.write(f"Bootstrap CI (crisis) & [{ci_lower:.2f}, {ci_upper:.2f}] & -- & "
            f"Gap $>$ 0 at 95\\%\\\\\n")
    za_sig = za_result['Significance'] if za_result['Significance'] else '--'
    za_t = f"{za_result['ZA t-stat']:.2f}" if za_result['ZA t-stat'] else '--'
    f.write(f"ZA break (gap) & $t={za_t}$ & -- & "
            f"Break: {za_result['Break Year']} ({za_sig})\\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")
print("Saved: gender_gap_tests.tex")

print("\n✓ Gender analysis complete.")
