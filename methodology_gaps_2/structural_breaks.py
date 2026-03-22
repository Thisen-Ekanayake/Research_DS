"""
structural_breaks.py
====================
Zivot-Andrews (Model C) + Bai-Perron PELT structural break tests.
Updated to include remittances, agricultural output index, informal
employment, part-time employment, and discouraged workers alongside
the original 6 series.

Reads:  master_dataset.csv  (output of build_master.py + exchange_rate_backfill.py)
Writes: za_results.csv, bp_results.csv, structural_breaks.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ruptures as rpt
import warnings
warnings.filterwarnings('ignore')

MASTER = '/mnt/user-data/outputs/master_dataset.csv'

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
df = pd.read_csv(MASTER)
df['year'] = df['year'].astype(int)
df = df[df['year'].between(2015, 2024)].set_index('year')

print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns\n")

# ── 2. ZIVOT-ANDREWS ──────────────────────────────────────────────────────────
def zivot_andrews(series, name):
    y = series.dropna().values
    n = len(y)
    results = []
    trim = max(2, int(0.15 * n))
    for tb in range(trim, n - trim):
        du = np.array([1 if t >= tb else 0 for t in range(n)])
        dt = np.array([t - tb if t >= tb else 0 for t in range(n)])
        dy = np.diff(y)
        y_lag = y[:-1]
        X = np.column_stack([np.ones(len(dy)), y_lag, du[1:], dt[1:]])
        try:
            coef, _, _, _ = np.linalg.lstsq(X, dy, rcond=None)
            resid = dy - X @ coef
            sigma2 = np.sum(resid**2) / max(len(dy) - X.shape[1], 1)
            cov = sigma2 * np.linalg.inv(X.T @ X)
            t_stat = coef[1] / np.sqrt(cov[1, 1])
            results.append((tb, t_stat))
        except Exception:
            pass
    if not results:
        return {'Series': name, 'Break Year': None, 'ZA t-stat': np.nan,
                'CV 1%': -5.57, 'CV 5%': -5.08, 'CV 10%': -4.82,
                'Significance': '', 'Reject Unit Root': False}
    results.sort(key=lambda x: x[1])
    tb_idx, min_t = results[0]
    break_year = series.dropna().index[tb_idx]
    cv = {0.01: -5.57, 0.05: -5.08, 0.10: -4.82}
    sig = ('***' if min_t < cv[0.01] else
           '**'  if min_t < cv[0.05] else
           '*'   if min_t < cv[0.10] else '')
    return {'Series': name, 'Break Year': break_year,
            'ZA t-stat': round(min_t, 4),
            'CV 1%': cv[0.01], 'CV 5%': cv[0.05], 'CV 10%': cv[0.10],
            'Significance': sig, 'Reject Unit Root': min_t < cv[0.10]}

# ── 3. BAI-PERRON ─────────────────────────────────────────────────────────────
def bai_perron(series, name, max_breaks=2):
    y = series.dropna().values
    years = series.dropna().index.tolist()
    n = len(y)
    if n < 6:
        return {'Series': name, 'PELT Breaks': [], 'BinSeg Breaks': [], 'N PELT': 0, 'N BinSeg': 0}
    model = rpt.Pelt(model='rbf', min_size=2, jump=1).fit(y)
    breaks_idx = model.predict(pen=1.5)
    break_years = [years[i-1] for i in breaks_idx if i < n]
    n_bkps = min(max_breaks, max(1, n // 4))
    try:
        model2 = rpt.Binseg(model='l2', min_size=2).fit(y)
        breaks2 = model2.predict(n_bkps=n_bkps)
        break_years2 = [years[i-1] for i in breaks2 if i < n]
    except Exception:
        break_years2 = []
    return {'Series': name, 'PELT Breaks': break_years, 'BinSeg Breaks': break_years2,
            'N PELT': len(break_years), 'N BinSeg': len(break_years2)}

# ── 4. SERIES TO TEST ─────────────────────────────────────────────────────────
# Original 6 + 5 new variables
series_map = {
    # Original
    'Underemployment rate':       'underemployment_total',
    'GDP growth':                 'gdp_growth_pct',
    'Inflation':                  'inflation_cpi_pct',
    'Youth LFPR':                 'youth_lfpr_pct',
    'Informal employment':        'informal_emp_pct',
    'Exchange rate':              'exchange_rate_lkr_usd',
    # New variables
    'Remittances (USD)':          'remittances_usd',
    'Agricultural output index':  'agri_output_index',
    'Part-time employment':       'parttime_emp_pct',
    'Discouraged workers':        'discouraged_seekers_n',
    'TRU female':                 'tru_female',
}

za_results, bp_results = [], []
for name, col in series_map.items():
    if col not in df.columns:
        print(f"SKIP (not in dataset): {name}")
        continue
    s = df[col].dropna()
    if len(s) < 6:
        print(f"SKIP (n<6): {name}")
        continue
    r = zivot_andrews(s, name)
    za_results.append(r)
    r2 = bai_perron(s, name)
    bp_results.append(r2)
    print(f"ZA | {name:<30} break={r['Break Year']}  t={r['ZA t-stat']:>7}  {r['Significance']}")
    print(f"BP | {name:<30} PELT={r2['PELT Breaks']}  BinSeg={r2['BinSeg Breaks']}")

za_df = pd.DataFrame(za_results)
bp_df = pd.DataFrame(bp_results)

# ── 5. PLOT ───────────────────────────────────────────────────────────────────
n_series = len(za_results)
ncols = 3
nrows = int(np.ceil(n_series / ncols))
colors = plt.cm.tab10(np.linspace(0, 1, n_series))

fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4))
fig.suptitle(
    'Zivot-Andrews & Bai-Perron Structural Break Analysis\nSri Lanka Labour Market (2015–2024) — Extended Variable Set',
    fontsize=13, fontweight='bold', y=0.99
)
axes_flat = axes.flat

for ax, (name, col), color, za_r, bp_r in zip(
        axes_flat, series_map.items(), colors, za_results, bp_results):
    s = df[col].dropna()
    years = s.index.tolist()
    ax.plot(years, s.values, color=color, linewidth=2, marker='o', markersize=4)
    if za_r['Break Year'] is not None:
        ax.axvline(za_r['Break Year'], color='red', linestyle='--', linewidth=1.5,
                   label=f"ZA {za_r['Break Year']} {za_r['Significance']}")
    for bp_yr in bp_r['PELT Breaks']:
        ax.axvline(bp_yr, color='orange', linestyle=':', linewidth=1.5, label=f"BP {bp_yr}")
    ax.axvspan(2021.5, 2022.5, alpha=0.1, color='red')
    ax.set_title(name, fontsize=9, fontweight='bold')
    ax.legend(fontsize=6.5, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], rotation=45, fontsize=6)

# Hide any unused subplots
for ax in list(axes_flat)[len(za_results):]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/structural_breaks.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved: structural_breaks.png")

# ── 6. SAVE ───────────────────────────────────────────────────────────────────
print("\n=== ZIVOT-ANDREWS RESULTS ===")
print(za_df[['Series','Break Year','ZA t-stat','CV 5%','Significance','Reject Unit Root']].to_string(index=False))
print("\n=== BAI-PERRON RESULTS ===")
print(bp_df[['Series','PELT Breaks','BinSeg Breaks']].to_string(index=False))

za_df.to_csv('/mnt/user-data/outputs/za_results.csv', index=False)
bp_df.to_csv('/mnt/user-data/outputs/bp_results.csv', index=False)
print("\nSaved: za_results.csv, bp_results.csv, structural_breaks.png")
