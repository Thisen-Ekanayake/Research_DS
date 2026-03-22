"""
run_shap_analysis.py
====================
SHAP analysis for Sri Lanka underemployment drivers.
Updated to:
  - Use master_dataset.csv (underemployment as target, not unemployment)
  - Include 5 new variables: remittances, agri output, part-time emp,
    discouraged workers, exchange rate (real, no imputation)
  - Add force plot and waterfall chart for 2022 crisis peak (RQ2 missing output)
  - Save all plots to Visualizations/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
import shap
from sklearn.metrics import r2_score
from pathlib import Path

MASTER = '/mnt/user-data/outputs/master_dataset.csv'
OUTDIR = Path('/mnt/user-data/outputs')
OUTDIR.mkdir(exist_ok=True)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
df = pd.read_csv(MASTER)
df['year'] = df['year'].astype(int)
df = df[df['year'].between(2015, 2024)].sort_values('year').reset_index(drop=True)

TARGET = 'underemployment_total'

FEATURES = [
    'gdp_growth_pct',
    'inflation_cpi_pct',
    'exchange_rate_lkr_usd',
    'youth_lfpr_pct',
    'informal_emp_pct',
    # New variables
    'remittances_usd',
    'agri_output_index',
    'parttime_emp_pct',
    'discouraged_seekers_n',
]
# Keep only features present in dataset
FEATURES = [f for f in FEATURES if f in df.columns]

print(f"Target: {TARGET}")
print(f"Features ({len(FEATURES)}): {FEATURES}")
print(f"n = {len(df)}")

X = df[FEATURES].copy()
y = df[TARGET].copy()

# Impute any remaining NaNs column-wise (mean)
X = X.fillna(X.mean())

# ── 2. TRAIN XGBOOST (temporal 80/20 split — time order preserved) ───────────
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = xgb.XGBRegressor(
    n_estimators=100, max_depth=3, learning_rate=0.1,
    subsample=0.8, random_state=42, verbosity=0
)
model.fit(X_train, y_train)
r2_test = r2_score(y_test, model.predict(X_test))
r2_full = r2_score(y, model.predict(X))
print(f"\nXGBoost R² (test): {r2_test:.4f}")
print(f"XGBoost R² (full): {r2_full:.4f}")

# ── 3. SHAP VALUES ────────────────────────────────────────────────────────────
explainer   = shap.TreeExplainer(model)
shap_vals   = explainer(X)          # Explanation object (for waterfall/force)
shap_matrix = explainer.shap_values(X)   # numpy array (for summary/beeswarm)

mean_abs = np.abs(shap_matrix).mean(axis=0)
importance_df = pd.DataFrame({'feature': FEATURES, 'mean_abs_shap': mean_abs})\
    .sort_values('mean_abs_shap', ascending=False)
print("\nSHAP feature importance:")
print(importance_df.to_string(index=False))

# ── 4. BEESWARM + BAR SUMMARY ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('SHAP Feature Association — Sri Lanka Underemployment (2015–2024)',
             fontsize=13, fontweight='bold')

plt.sca(axes[0])
shap.summary_plot(shap_matrix, X, plot_type='dot', show=False)
axes[0].set_title('Beeswarm (association direction & magnitude)', fontsize=10)

plt.sca(axes[1])
shap.summary_plot(shap_matrix, X, plot_type='bar', show=False)
axes[1].set_title('Mean |SHAP| feature importance', fontsize=10)

plt.tight_layout()
plt.savefig(OUTDIR / 'shap_summary_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: shap_summary_plots.png")

# ── 5. WATERFALL CHART — 2022 crisis peak ─────────────────────────────────────
# 2022 is the crisis peak year — highest underemployment + structural break
idx_2022 = df[df['year'] == 2022].index
if len(idx_2022) > 0:
    i = idx_2022[0]
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(shap_vals[i], max_display=len(FEATURES), show=False)
    ax = plt.gca()
    ax.set_title(f'SHAP Waterfall — 2022 Crisis Peak (Underemployment = {y.iloc[i]:.1f}%)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTDIR / 'shap_waterfall_2022.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: shap_waterfall_2022.png")

# ── 6. FORCE PLOT — 2022 ──────────────────────────────────────────────────────
if len(idx_2022) > 0:
    i = idx_2022[0]
    base_val = explainer.expected_value
    if isinstance(base_val, np.ndarray):
        base_val = float(base_val[0])
    fig, ax = plt.subplots(figsize=(14, 3))
    shap.force_plot(
        base_val,
        shap_matrix[i],
        X.iloc[i],
        matplotlib=True,
        show=False
    )
    plt.title(f'SHAP Force Plot — 2022 (Underemployment = {y.iloc[i]:.1f}%)',
              fontsize=10, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTDIR / 'shap_force_2022.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: shap_force_2022.png")

# ── 7. DEPENDENCE PLOTS — top 4 features ──────────────────────────────────────
top4 = importance_df.head(4)['feature'].tolist()
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, feat in zip(axes.flat, top4):
    plt.sca(ax)
    shap.dependence_plot(feat, shap_matrix, X, ax=ax, show=False)
    ax.set_title(f'SHAP dependence: {feat}', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTDIR / 'shap_dependence_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: shap_dependence_plots.png")

print(f"\nAll SHAP outputs saved to {OUTDIR}")
print(f"XGBoost R² (full dataset): {r2_full:.4f}")
