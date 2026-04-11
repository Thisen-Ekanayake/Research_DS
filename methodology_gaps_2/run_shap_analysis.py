"""
run_shap_analysis.py  (v2 — methodologically corrected)
=========================================================
Fixes applied:
  [2] Bootstrap CIs (500 iter) on mean |SHAP| — honest uncertainty on feature rankings
  [3] LOOCV + Ridge replaces meaningless 80/20 XGBoost split on n=10
  [4] Sub-period waterfall/force plots retained; sub-period SHAP ranking
      explicitly caveated as illustrative (n<5 → no inferential claims)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import shap
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).resolve().parent
MASTER = BASE.parent / 'DataLoader' / 'master_dataset.csv'
OUTDIR = BASE.parent / 'output'
OUTDIR.mkdir(exist_ok=True)

# ── 1. LOAD & CLEAN DATA ──────────────────────────────────────────────────────
df_raw = pd.read_csv(MASTER)
df_raw.columns = df_raw.columns.str.strip()
for col in df_raw.columns:
    if col != 'Year':
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

df = df_raw[df_raw['Year'].between(2015, 2024)].sort_values('Year').reset_index(drop=True)

TARGET = 'Underemployment_Rate'
FEATURES = [
    'GDP_Growth_Rate',
    'Inflation_Rate',
    'Exchange_Rate_LKR_USD',
    'Youth_LFPR_15_24',
    'Informal_Pct',
    'Remit_Personal_remittances_received_current_US$',
    'AgriProdIdx_Agriculture',
]
LABELS = {
    'GDP_Growth_Rate':    'GDP Growth',
    'Inflation_Rate':     'Inflation',
    'Exchange_Rate_LKR_USD': 'Exchange Rate',
    'Youth_LFPR_15_24':  'Youth LFPR',
    'Informal_Pct':      'Informal Emp.',
    'Remit_Personal_remittances_received_current_US$': 'Remittances',
    'AgriProdIdx_Agriculture': 'Agri. Output',
}
FEATURES = [f for f in FEATURES if f in df.columns]

print(f"n = {len(df)}  (years {df['Year'].min()}–{df['Year'].max()})")
print(f"Target : {TARGET}")
print(f"Features ({len(FEATURES)}): {[LABELS.get(f, f) for f in FEATURES]}")
print()

X = df[FEATURES].fillna(df[FEATURES].mean())
y = df[TARGET].astype(float)

# Scale — Ridge is scale-sensitive
scaler = StandardScaler()
X_sc = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES, index=df.index)

# ── 2. MODEL: RIDGE + LOOCV ───────────────────────────────────────────────────
# With n=10 an 80/20 split yields 2 test points — statistically meaningless.
# LOOCV is the only valid evaluation strategy at this sample size.

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# RidgeCV uses its own efficient LOOCV to select alpha
ridge_cv = RidgeCV(alphas=ALPHAS, scoring='neg_mean_squared_error')
ridge_cv.fit(X_sc, y)
best_alpha = ridge_cv.alpha_
print(f"Ridge α selected by LOOCV: {best_alpha}")

# Full LOOCV pass for reported metrics
loo = LeaveOneOut()
y_pred_loo = np.zeros(len(y))
for train_idx, test_idx in loo.split(X_sc):
    m = Ridge(alpha=best_alpha)
    m.fit(X_sc.iloc[train_idx], y.iloc[train_idx])
    y_pred_loo[test_idx] = m.predict(X_sc.iloc[test_idx])

r2_loo   = r2_score(y, y_pred_loo)
mae_loo  = mean_absolute_error(y, y_pred_loo)
rmse_loo = np.sqrt(mean_squared_error(y, y_pred_loo))

print(f"\nLOOCV performance (n={len(y)}, Ridge α={best_alpha}):")
print(f"  R²   = {r2_loo:.3f}")
print(f"  MAE  = {mae_loo:.4f} pp")
print(f"  RMSE = {rmse_loo:.4f} pp")
print("  NOTE: All metrics from LOOCV — no held-out annual split used.")

# Final model on all data (for SHAP explanation)
final_model = Ridge(alpha=best_alpha)
final_model.fit(X_sc, y)

# ── 3. SHAP ON FULL DATASET ───────────────────────────────────────────────────
# LinearExplainer is exact for linear models (no approximation needed).
explainer   = shap.LinearExplainer(final_model, X_sc)
shap_matrix = explainer.shap_values(X_sc)   # (n, p) numpy array
shap_vals   = explainer(X_sc)               # Explanation object for waterfall/force

mean_abs_shap = np.abs(shap_matrix).mean(axis=0)

# ── 4. BOOTSTRAP CIs ON mean|SHAP| — 500 iterations ─────────────────────────
# Bootstrap resamples years (with replacement) to quantify ranking uncertainty.
# Wide / overlapping CIs mean we cannot claim a definitive feature ordering.

N_BOOT = 500
rng = np.random.default_rng(42)
boot_mean_abs = np.zeros((N_BOOT, len(FEATURES)))

for b in range(N_BOOT):
    idx = rng.choice(len(X_sc), size=len(X_sc), replace=True)
    X_b = X_sc.iloc[idx].reset_index(drop=True)
    y_b = y.iloc[idx].reset_index(drop=True)
    m_b = Ridge(alpha=best_alpha).fit(X_b, y_b)
    exp_b = shap.LinearExplainer(m_b, X_b)
    sv_b  = exp_b.shap_values(X_sc)          # explain full (fixed) dataset
    boot_mean_abs[b] = np.abs(sv_b).mean(axis=0)

ci_lo = np.percentile(boot_mean_abs, 2.5,  axis=0)
ci_hi = np.percentile(boot_mean_abs, 97.5, axis=0)

importance_df = pd.DataFrame({
    'feature':       FEATURES,
    'label':         [LABELS.get(f, f) for f in FEATURES],
    'mean_abs_shap': mean_abs_shap,
    'ci_lo':         ci_lo,
    'ci_hi':         ci_hi,
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
importance_df['rank'] = range(1, len(importance_df) + 1)

# Flag pairs whose CIs overlap — we cannot claim a definitive rank difference
importance_df['ci_overlaps_next'] = False
for i in range(len(importance_df) - 1):
    if importance_df.loc[i, 'ci_lo'] < importance_df.loc[i + 1, 'ci_hi']:
        importance_df.loc[i,     'ci_overlaps_next'] = True
        importance_df.loc[i + 1, 'ci_overlaps_next'] = True

print("\nSHAP feature importance with 95% bootstrap CIs (n_boot=500):")
print(importance_df[['rank', 'label', 'mean_abs_shap', 'ci_lo', 'ci_hi',
                      'ci_overlaps_next']].to_string(index=False))

n_overlap = importance_df['ci_overlaps_next'].sum()
if n_overlap > 0:
    print(f"\n⚠  {n_overlap} feature(s) have overlapping CIs with adjacent rank —"
          " definitive ordering cannot be claimed for those pairs.")

# ── 5. BEESWARM + BOOTSTRAP CI BAR ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('SHAP Analysis — Sri Lanka Underemployment (2015–2024)\n'
             '(Ridge regression, LOOCV evaluation, n=10)',
             fontsize=12, fontweight='bold')

plt.sca(axes[0])
shap.summary_plot(shap_matrix, X_sc, feature_names=[LABELS.get(f, f) for f in FEATURES],
                  plot_type='dot', show=False)
axes[0].set_title('Beeswarm: direction & magnitude', fontsize=10)

# Bootstrap CI bar chart
ax2 = axes[1]
y_pos = np.arange(len(importance_df))
colors = ['#EF4444' if ov else '#2563EB' for ov in importance_df['ci_overlaps_next']]
ax2.barh(y_pos, importance_df['mean_abs_shap'], color=colors, alpha=0.75, label='Mean |SHAP|')
ax2.errorbar(importance_df['mean_abs_shap'], y_pos,
             xerr=[importance_df['mean_abs_shap'] - importance_df['ci_lo'],
                   importance_df['ci_hi'] - importance_df['mean_abs_shap']],
             fmt='none', color='black', capsize=4, linewidth=1.2)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(importance_df['label'])
ax2.invert_yaxis()
ax2.set_xlabel('Mean |SHAP value|', fontsize=10)
ax2.set_title('Feature Importance with 95% Bootstrap CIs\n'
              '(Red = CI overlaps adjacent rank)', fontsize=10)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTDIR / 'shap_summary_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: shap_summary_plots.png")

# ── 6. WATERFALL — 2022 crisis peak ──────────────────────────────────────────
idx_2022 = df[df['Year'] == 2022].index
if len(idx_2022) > 0:
    i = idx_2022[0]
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(shap_vals[i], max_display=len(FEATURES), show=False)
    ax = plt.gca()
    ax.set_title(
        f'SHAP Waterfall — 2022 Crisis Peak (Underemployment = {y.iloc[i]:.1f}%)\n'
        'Illustrative: single-year explanation, not an inferential result (n=10)',
        fontsize=10, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(OUTDIR / 'shap_waterfall_2022.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: shap_waterfall_2022.png")

# ── 7. FORCE PLOT — 2022 ─────────────────────────────────────────────────────
if len(idx_2022) > 0:
    i = idx_2022[0]
    base_val = float(explainer.expected_value
                     if not isinstance(explainer.expected_value, np.ndarray)
                     else explainer.expected_value[0])
    fig, ax = plt.subplots(figsize=(14, 3))
    shap.force_plot(base_val, shap_matrix[i], X_sc.iloc[i],
                    feature_names=[LABELS.get(f, f) for f in FEATURES],
                    matplotlib=True, show=False)
    plt.title(f'SHAP Force Plot — 2022 (Underemployment = {y.iloc[i]:.1f}%)',
              fontsize=10, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTDIR / 'shap_force_2022.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: shap_force_2022.png")

# ── 8. DEPENDENCE PLOTS — top 4 features ─────────────────────────────────────
top4 = importance_df.head(4)['feature'].tolist()
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, feat in zip(axes.flat, top4):
    feat_idx = FEATURES.index(feat)   # dependence_plot needs int index when feature_names differ
    plt.sca(ax)
    shap.dependence_plot(feat_idx, shap_matrix, X_sc,
                         feature_names=[LABELS.get(f, f) for f in FEATURES],
                         ax=ax, show=False)
    ax.set_title(f'SHAP dependence: {LABELS.get(feat, feat)}', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTDIR / 'shap_dependence_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: shap_dependence_plots.png")

# ── 9. SUMMARY PRINTOUT ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SHAP ANALYSIS RESULTS")
print("=" * 60)
print(f"\nModel          : Ridge regression (α={best_alpha})")
print(f"Validation     : Leave-One-Out CV (n={len(y)})")
print(f"LOOCV R²       : {r2_loo:.3f}")
print(f"LOOCV MAE      : {mae_loo:.4f} pp")
print(f"LOOCV RMSE     : {rmse_loo:.4f} pp")
print(f"Bootstrap iters: {N_BOOT}")
print()
print("Feature ranking (point estimate ± 95% CI):")
for _, row in importance_df.iterrows():
    overlap_flag = ' ← CI overlaps adjacent rank' if row['ci_overlaps_next'] else ''
    print(f"  {int(row['rank']):2d}. {row['label']:20s}  "
          f"{row['mean_abs_shap']:.4f}  [{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]"
          f"{overlap_flag}")

print(f"\nAll outputs saved to: {OUTDIR}")
