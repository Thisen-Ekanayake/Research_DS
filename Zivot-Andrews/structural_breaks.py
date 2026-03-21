import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ruptures as rpt
import warnings
warnings.filterwarnings('ignore')

# ── 1. LOAD COMBINED DATASET ──────────────────────────────────────────────────
base = pd.read_csv('Zivot-Andrews/sri_lanka_labour_macro_combined.csv')
print(f"Loaded: {base.shape[0]} rows x {base.shape[1]} columns")
print(base[['year','underemployment_total_pct','gdp_growth_pct','inflation_pct',
            'agri_output_index','services_employment_share_pct',
            'youth_lfpr_female_pct']].to_string(index=False))

# ── 2. ZIVOT-ANDREWS TEST ─────────────────────────────────────────────────────
def zivot_andrews(series, name):
    """
    Zivot-Andrews unit root test with endogenous structural break (Model C).
    Searches all candidate breakpoints; selects the one with the most negative
    t-statistic on the lagged level — the point where stationarity is most
    plausible conditional on a break.
    """
    y = series.dropna().values
    n = len(y)
    results = []
    trim = int(0.15 * n)

    for tb in range(trim, n - trim):
        du = np.array([1 if t >= tb else 0 for t in range(n)])
        dt = np.array([t - tb if t >= tb else 0 for t in range(n)])
        dy = np.diff(y)
        y_lag = y[:-1]
        du_t = du[1:]
        dt_t = dt[1:]

        X = np.column_stack([np.ones(len(dy)), y_lag, du_t, dt_t])
        try:
            coef, _, _, _ = np.linalg.lstsq(X, dy, rcond=None)
            resid = dy - X @ coef
            sigma2 = np.sum(resid**2) / (len(dy) - X.shape[1])
            cov = sigma2 * np.linalg.inv(X.T @ X)
            t_stat = coef[1] / np.sqrt(cov[1, 1])
            results.append((tb, t_stat))
        except:
            pass

    results.sort(key=lambda x: x[1])
    tb_idx, min_t = results[0]
    break_year = series.dropna().index[tb_idx]

    cv = {0.01: -5.57, 0.05: -5.08, 0.10: -4.82}
    sig = "***" if min_t < cv[0.01] else ("**" if min_t < cv[0.05] else ("*" if min_t < cv[0.10] else ""))

    return {
        'Series': name,
        'Break Year': break_year,
        'ZA t-stat': round(min_t, 4),
        'CV 1%': cv[0.01],
        'CV 5%': cv[0.05],
        'CV 10%': cv[0.10],
        'Significance': sig,
        'Reject Unit Root': min_t < cv[0.10]
    }

# ── 3. BAI-PERRON TEST ────────────────────────────────────────────────────────
def bai_perron(series, name, max_breaks=2):
    """
    Bai-Perron multiple structural break detection.
    PELT finds the globally optimal breakpoints; BinSeg is used as a cross-check.
    """
    y = series.dropna().values
    years = series.dropna().index.tolist()
    n = len(y)

    if n < 6:
        return {'Series': name, 'PELT Breaks': [], 'BinSeg Breaks': [], 'N PELT': 0, 'N BinSeg': 0}

    model = rpt.Pelt(model="rbf", min_size=2, jump=1).fit(y)
    breaks_idx = model.predict(pen=1.5)
    break_years = [years[i-1] for i in breaks_idx if i < n]

    n_bkps = min(max_breaks, max(1, n // 4))
    try:
        model2 = rpt.Binseg(model="l2", min_size=2).fit(y)
        breaks2 = model2.predict(n_bkps=n_bkps)
        break_years2 = [years[i-1] for i in breaks2 if i < n]
    except Exception:
        break_years2 = []

    return {
        'Series': name,
        'PELT Breaks': break_years,
        'BinSeg Breaks': break_years2,
        'N PELT': len(break_years),
        'N BinSeg': len(break_years2),
    }

# ── 4. RUN TESTS ──────────────────────────────────────────────────────────────
# Map display names to combined dataset column names
series_map = {
    'Underemployment Rate':      'underemployment_total_pct',
    'GDP Growth':                'gdp_growth_pct',
    'Inflation':                 'inflation_pct',
    'Agricultural Output Index': 'agri_output_index',
    'Services Employment Share': 'services_employment_share_pct',
    'Youth LFPR':                'youth_lfpr_female_pct',
}

indexed = base.set_index('year')

za_results, bp_results = [], []
for name, col in series_map.items():
    s = indexed[col].dropna()
    try:
        r = zivot_andrews(s, name)
        za_results.append(r)
        print(f"ZA | {name}: break={r['Break Year']}, t={r['ZA t-stat']} {r['Significance']}")
    except Exception as e:
        print(f"ZA | {name}: ERROR {e}")
    try:
        r2 = bai_perron(s, name)
        bp_results.append(r2)
        print(f"BP | {name}: PELT={r2['PELT Breaks']}, BinSeg={r2['BinSeg Breaks']}")
    except Exception as e:
        print(f"BP | {name}: ERROR {e}")

za_df = pd.DataFrame(za_results)
bp_df = pd.DataFrame(bp_results)

# ── 5. PLOT ───────────────────────────────────────────────────────────────────
colors = ['#2563EB','#DC2626','#16A34A','#D97706','#7C3AED','#0891B2']
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('Zivot-Andrews & Bai-Perron Structural Break Analysis\nSri Lanka Labour Market (2015-2024)',
             fontsize=14, fontweight='bold', y=0.98)

plot_titles = {
    'Underemployment Rate':      'Underemployment\nRate (%)',
    'GDP Growth':                'GDP Growth\n(%)',
    'Inflation':                 'Inflation\n(%)',
    'Agricultural Output Index': 'Agricultural\nOutput Index',
    'Services Employment Share': 'Services\nShare (%)',
    'Youth LFPR':                'Youth LFPR\n(%)',
}

for ax, (name, col), color, za_r, bp_r in zip(
        axes.flat, series_map.items(), colors, za_results, bp_results):

    s = indexed[col].dropna()
    years = s.index.tolist()
    vals = s.values

    ax.plot(years, vals, color=color, linewidth=2, marker='o', markersize=5, label='Series')

    za_yr = za_r['Break Year']
    ax.axvline(za_yr, color='red', linestyle='--', linewidth=1.5,
               label=f"ZA: {za_yr} (t={za_r['ZA t-stat']}){za_r['Significance']}")

    for i, bp_yr in enumerate(bp_r['PELT Breaks']):
        ax.axvline(bp_yr, color='orange', linestyle=':', linewidth=1.5,
                   label=f"BP: {bp_yr}" if i == 0 else f"BP: {bp_yr}")

    ax.axvspan(2022, 2023, alpha=0.08, color='red', label='2022 Crisis')

    ax.set_title(plot_titles[name], fontsize=10, fontweight='bold')
    ax.set_xlabel('Year', fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6.5, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45)

plt.tight_layout()
plt.savefig('structural_breaks.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved: structural_breaks.png")

# ── 6. PRINT & SAVE RESULTS ───────────────────────────────────────────────────
print("\n=== ZIVOT-ANDREWS RESULTS ===")
print(za_df[['Series','Break Year','ZA t-stat','CV 5%','Significance','Reject Unit Root']].to_string(index=False))

print("\n=== BAI-PERRON RESULTS ===")
print(bp_df[['Series','PELT Breaks','BinSeg Breaks']].to_string(index=False))

za_df.to_csv('za_results.csv', index=False)
bp_df.to_csv('bp_results.csv', index=False)
print("\nResults saved: za_results.csv, bp_results.csv")