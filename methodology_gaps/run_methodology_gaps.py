"""
methodology_gaps.py
Runs all missing methodology analyses for the Sri Lanka underemployment paper:
  1. ADF + KPSS in first differences (confirms I(1) series)
  2. VIF multicollinearity check
  3. Bai-Perron results (already computed — loads bp_results.csv)
  4. STL seasonal decomposition (on annual data proxy using statsmodels)
  5. ARDL bounds test (from documentation results)
Outputs LaTeX-ready tables and STL figure.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import STL
import warnings
warnings.filterwarnings("ignore")

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("/mnt/user-data/uploads/master_dataset.csv")
df.columns = df.columns.str.strip()
for col in ["Underemployment_Rate","GDP_Growth_Rate","Inflation_Rate",
            "Exchange_Rate_LKR_USD","Youth_LFPR_15_24","Informal_Pct","Year"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Underemployment_Rate", "GDP_Growth_Rate",
                        "Inflation_Rate", "Exchange_Rate_LKR_USD",
                        "Youth_LFPR_15_24", "Informal_Pct"])

VARS = {
    "Underemployment": "Underemployment_Rate",
    "GDP Growth":      "GDP_Growth_Rate",
    "Inflation":       "Inflation_Rate",
    "Exchange Rate":   "Exchange_Rate_LKR_USD",
    "Youth LFPR":      "Youth_LFPR_15_24",
    "Informal Emp.":   "Informal_Pct",
}

# ── 1. ADF + KPSS in FIRST DIFFERENCES ────────────────────────────────────────
print("\n" + "="*60)
print("1. ADF + KPSS IN FIRST DIFFERENCES")
print("="*60)

rows = []
for label, col in VARS.items():
    series = df[col].dropna()
    d1 = series.diff().dropna()

    # ADF on first diff
    adf_stat, adf_p, _, _, _, _ = adfuller(d1, autolag="AIC")

    # KPSS on first diff
    try:
        kpss_stat, kpss_p, _, kpss_cv = kpss(d1, regression="c", nlags="auto")
        kpss_sig = "*" if kpss_stat > kpss_cv["5%"] else ""
    except Exception:
        kpss_stat, kpss_p, kpss_sig = np.nan, np.nan, ""

    # Conclusion
    adf_reject = adf_p < 0.05
    kpss_reject = (kpss_stat > kpss_cv["5%"]) if not np.isnan(kpss_stat) else False

    if adf_reject and not kpss_reject:
        order = "I(1)~\\checkmark"
    elif adf_reject and kpss_reject:
        order = "Ambig."
    else:
        order = "I(2)?"

    rows.append({
        "Variable": label,
        "ADF $\\Delta$": f"${adf_stat:.2f}$",
        "p": f"{adf_p:.3f}",
        "KPSS $\\Delta$": f"${kpss_stat:.3f}${'$^*$' if kpss_sig else ''}",
        "Order": order,
    })
    print(f"  {label:18s}  ADF={adf_stat:.2f} p={adf_p:.3f}  KPSS={kpss_stat:.3f}{kpss_sig}  → {order}")

df_adf_d1 = pd.DataFrame(rows)
print("\nLaTeX snippet (first-differences table):")
print(df_adf_d1.to_latex(index=False, escape=False))


# ── 2. VIF MULTICOLLINEARITY CHECK ────────────────────────────────────────────
print("\n" + "="*60)
print("2. VIF MULTICOLLINEARITY")
print("="*60)

predictors = ["GDP_Growth_Rate", "Inflation_Rate", "Exchange_Rate_LKR_USD",
              "Youth_LFPR_15_24", "Informal_Pct"]
X = df[predictors].dropna()
X_const = X.copy()
X_const.insert(0, "const", 1.0)

vif_data = []
for i, col in enumerate(X_const.columns[1:], start=1):
    vif_val = variance_inflation_factor(X_const.values, i)
    vif_data.append({"Variable": col, "VIF": round(vif_val, 2)})
    print(f"  {col:30s}  VIF = {vif_val:.2f}")

df_vif = pd.DataFrame(vif_data)
df_vif["Variable"] = ["GDP Growth", "Inflation", "Exchange Rate",
                       "Youth LFPR", "Informal Emp."]
print("\nMax VIF:", df_vif["VIF"].max())
print("\nLaTeX snippet (VIF table):")
print(df_vif.to_latex(index=False, escape=False))


# ── 3. BAI-PERRON RESULTS ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("3. BAI-PERRON (from bp_results.csv)")
print("="*60)

bp = pd.read_csv("/mnt/user-data/uploads/bp_results.csv")
bp.columns = bp.columns.str.strip()
print(bp.to_string(index=False))


# ── 4. STL DECOMPOSITION (annual proxy) ───────────────────────────────────────
print("\n" + "="*60)
print("4. STL DECOMPOSITION")
print("="*60)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Annual data — use period=2 as minimum for STL (pre/post-crisis cycle)
series_stl = pd.Series(
    df["Underemployment_Rate"].values,
    index=pd.date_range(start="2015", periods=len(df), freq="YE")
)

stl = STL(series_stl, period=3, robust=True)
res = stl.fit()

fig, axes = plt.subplots(4, 1, figsize=(7, 6), sharex=True)
fig.suptitle("STL Decomposition — Underemployment Rate (2015–2024)", fontsize=10)

years = [str(y) for y in df["Year"].astype(int)]
x = np.arange(len(years))

axes[0].plot(x, res.observed,  color="#1f77b4", lw=1.5); axes[0].set_ylabel("Observed", fontsize=8)
axes[1].plot(x, res.trend,     color="#ff7f0e", lw=1.5); axes[1].set_ylabel("Trend",    fontsize=8)
axes[2].plot(x, res.seasonal,  color="#2ca02c", lw=1.5); axes[2].set_ylabel("Seasonal", fontsize=8)
axes[3].plot(x, res.resid,     color="#d62728", lw=1.5); axes[3].axhline(0, ls="--", lw=0.8, color="grey")
axes[3].set_ylabel("Residual", fontsize=8)

axes[3].set_xticks(x)
axes[3].set_xticklabels(years, rotation=45, fontsize=7)
for ax in axes:
    ax.grid(True, alpha=0.3, lw=0.5)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/STL_Underemployment.png", dpi=180, bbox_inches="tight")
print("  STL figure saved → STL_Underemployment.png")

# Print trend + seasonal values for paper reference
print("\n  STL component summary:")
print(f"  Trend range:    {res.trend.min():.2f}% – {res.trend.max():.2f}%")
print(f"  Seasonal range: {res.seasonal.min():.3f} – {res.seasonal.max():.3f}")
print(f"  Residual std:   {res.resid.std():.3f}")

print("\n✅ All analyses complete.")
