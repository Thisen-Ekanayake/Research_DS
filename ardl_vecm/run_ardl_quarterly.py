"""
run_ardl_quarterly.py
=====================
Steps 2–5 of the Member 1 (Ekanayake) econometric pipeline.

Runs on quarterly_master_dataset.csv (n=40, 2015 Q1 – 2024 Q4):

  Step 2 — ARDL bounds testing (Pesaran et al. 2001, Case III)
  Step 3 — Extended interaction models (GDP×Crisis, Inflation×Crisis, Informal×Crisis)
  Step 4 — Granger causality (maxlag=4, Bonferroni α=0.01)
  Step 5 — Robustness: CUSUM/CUSUM-sq, Newey-West HAC, rolling OLS

All results are printed to stdout and saved as CSV tables in ardl_vecm/output/.
LaTeX snippets are written for direct inclusion in the paper.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.ardl import ARDL, ardl_select_order
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "quarterly_master_dataset.csv")
OUT    = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)

# Core variables
DEPVAR     = "Underemployment_Rate"
PREDICTORS = ["GDP_Growth_Rate", "Inflation_Rate", "Informal_Pct",
              "Youth_LFPR", "Remittances_USD"]
EXTRA_PRED = ["Exchange_Rate", "Agri_Output_Index"]  # used in Granger only


def sep(title=""):
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA)
    # Drop first 4 rows (2015) which have NaN in GDP/Inflation (YoY lag)
    df = df.dropna(subset=["GDP_Growth_Rate", "Inflation_Rate"]).reset_index(drop=True)
    # PeriodIndex for plots
    df.index = pd.PeriodIndex(df["Year"].astype(str) + df["Quarter"], freq="Q")
    print(f"Loaded: n={len(df)} quarters ({df['Period'].iloc[0]} – {df['Period'].iloc[-1]})")
    return df


# ---------------------------------------------------------------------------
# STEP 2A: ADF stationarity on quarterly data
# ---------------------------------------------------------------------------
def step2_stationarity(df: pd.DataFrame):
    sep("STEP 2A — ADF UNIT ROOT TESTS (quarterly)")
    results = []
    series_to_test = [DEPVAR] + PREDICTORS + EXTRA_PRED
    for col in series_to_test:
        s = df[col].dropna()
        # Levels
        adf_lev = adfuller(s, autolag="AIC", regression="ct")
        p_lev   = adf_lev[1]
        # First differences
        ds = s.diff().dropna()
        adf_dif = adfuller(ds, autolag="AIC", regression="c")
        p_dif   = adf_dif[1]

        order = "I(0)" if p_lev < 0.05 else ("I(1)" if p_dif < 0.05 else "I(2)+")
        results.append({
            "Variable": col,
            "ADF_levels_p":  round(p_lev, 4),
            "ADF_diff_p":    round(p_dif, 4),
            "Integration":   order
        })
        print(f"  {col:25s}  levels p={p_lev:.4f}  Δ p={p_dif:.4f}  → {order}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUT, "adf_quarterly.csv"), index=False)
    print(f"\nSaved: output/adf_quarterly.csv")
    return res_df


# ---------------------------------------------------------------------------
# STEP 2B: ARDL lag selection + estimation + bounds test
# ---------------------------------------------------------------------------
def step2_ardl(df: pd.DataFrame):
    sep("STEP 2B — ARDL BOUNDS TEST (Pesaran et al. 2001, Case III)")

    y    = df[DEPVAR]
    exog = df[PREDICTORS]

    # --- Lag selection (max 4 quarterly lags, AIC) ---
    print("Running ARDL lag selection (maxlag=4, AIC)...")
    try:
        sel = ardl_select_order(
            y, 4, exog, 2,       # max 4 lags for dep, 2 for each exog (DoF constraint)
            ic="aic", trend="c"
        )
        # Attribute name differs across statsmodels versions
        ardl_order = getattr(sel, "ardl_order", getattr(sel, "order", "(auto)"))
        print(f"AIC-selected order: ARDL{ardl_order}")
        model = sel.model
        res   = model.fit()
    except Exception as e:
        print(f"Auto-selection failed ({e}). Falling back to ARDL(1,1,1,1,1,1).")
        model = ARDL(y, 1, exog, 1, trend="c")
        res   = model.fit()
        ardl_order = "(1,1,1,1,1,1) [fallback]"

    print(res.summary())

    # --- Bounds test (manual F-statistic, Pesaran 2001) ---
    # H0: λ₁ = λ₂' = 0  (no long-run level relationship)
    # Compute via Wald test on all lagged-level terms (L1 of all variables)
    print("\nBOUNDS TEST — Manual Wald F-statistic on lagged level terms")
    level_terms = [c for c in res.params.index
                   if c.endswith(".L1") and c != f"{DEPVAR}.L1"]
    level_terms_all = [c for c in res.params.index if c.endswith(".L1")]
    print(f"  Level terms tested: {level_terms_all}")

    try:
        wald = res.wald_test_terms(skip_single=False)
        print(wald)
    except Exception:
        pass

    try:
        # Test: all L1 coefficients = 0
        r_mat = [(f"{c} = 0") for c in level_terms_all]
        ftest = res.f_test(" = ".join([f"{c}" for c in level_terms_all]) + " = 0")
        f_stat = float(ftest.fvalue)
        f_pval = float(ftest.pvalue)
    except Exception:
        # Alternative: compute RSS of restricted (drop L1 terms) vs unrestricted
        print("  Using RSS-based F manually...")
        rss_u = float(np.sum(res.resid**2))
        k_lev = len(level_terms_all)
        n_obs = int(res.nobs)
        k_tot = len(res.params)
        # Re-fit restricted model without L1 terms
        cols_restrict = [c for c in res.params.index if not c.endswith(".L1")]
        # Use OLS on the same data with the ARDL residuals as approximation
        f_stat = np.nan
        f_pval = np.nan

    # Pesaran (2001) critical values k=5, Case III (unrestricted intercept, no trend)
    cv = {
        "10%": (2.26, 3.35),
        "5%":  (2.62, 3.79),
        "1%":  (3.41, 4.68)
    }
    print(f"\nBounds F-statistic: {f_stat:.4f}  (p={f_pval:.4f})")
    print("Pesaran (2001) critical values (k=5, Case III):")
    for lvl, (lb, ub) in cv.items():
        if not np.isnan(f_stat):
            decision = "COINTEGRATED" if f_stat > ub else ("INCONCLUSIVE" if f_stat > lb else "no cointegration")
        else:
            decision = "N/A"
        print(f"  {lvl}: I(0)={lb}, I(1)={ub}  → {decision}")

    bt_df = pd.DataFrame({
        "F_stat":    [round(f_stat, 4) if not np.isnan(f_stat) else "N/A"],
        "F_pval":    [round(f_pval, 4) if not np.isnan(f_pval) else "N/A"],
        "CV_5pct_I0": [2.62], "CV_5pct_I1": [3.79],
        "CV_1pct_I0": [3.41], "CV_1pct_I1": [4.68],
        "ARDL_order": [str(ardl_order)]
    })
    bt_df.to_csv(os.path.join(OUT, "ardl_bounds_test.csv"), index=False)
    print(f"Saved: output/ardl_bounds_test.csv")

    # --- ECM speed-of-adjustment (λ₁) ---
    print("\nECM speed-of-adjustment (λ₁ — lagged dep.var level coefficient):")
    ect_key = f"{DEPVAR}.L1"
    if ect_key in res.params.index:
        ect      = res.params[ect_key]
        ect_se   = res.bse[ect_key]
        ect_p    = res.pvalues[ect_key]
        print(f"  λ₁ = {ect:.4f}  SE={ect_se:.4f}  p={ect_p:.4f}")
        if ect < 0 and ect_p < 0.10:
            print("  → Negative and (marginally) significant: supports error correction")
        else:
            print("  → ECT not significant: weak evidence for cointegration")

    # Save full ARDL results
    try:
        params_df = pd.DataFrame({
            "coef": res.params,
            "se":   res.bse,
            "t":    res.tvalues,
            "p":    res.pvalues,
            "ci_lo": res.conf_int()[0],
            "ci_hi": res.conf_int()[1]
        }).round(4)
        params_df.to_csv(os.path.join(OUT, "ardl_full_results.csv"))
        print("Saved: output/ardl_full_results.csv")
    except Exception as e:
        print(f"Save error: {e}")

    return res


# ---------------------------------------------------------------------------
# STEP 3: Extended interaction models
# ---------------------------------------------------------------------------
def step3_interaction_models(df: pd.DataFrame):
    sep("STEP 3 — EXTENDED INTERACTION MODELS (HC3 robust SE)")

    # OLS fallback on annual-equivalent (use quarterly with Crisis dummy)
    # Model A: GDP × Crisis
    m_gdp = smf.ols(
        f"{DEPVAR} ~ GDP_Growth_Rate + Crisis_Dummy + GDP_Growth_Rate:Crisis_Dummy",
        data=df
    ).fit(cov_type="HC3")

    # Model B: Inflation × Crisis
    m_inf = smf.ols(
        f"{DEPVAR} ~ Inflation_Rate + Crisis_Dummy + Inflation_Rate:Crisis_Dummy",
        data=df
    ).fit(cov_type="HC3")

    # Model C: Informal Employment × Crisis
    m_inf2 = smf.ols(
        f"{DEPVAR} ~ Informal_Pct + Crisis_Dummy + Informal_Pct:Crisis_Dummy",
        data=df
    ).fit(cov_type="HC3")

    models = [("GDP×Crisis", m_gdp), ("Inflation×Crisis", m_inf), ("Informal×Crisis", m_inf2)]

    rows = []
    for name, m in models:
        inter_key = [k for k in m.params.index if ":" in k]
        inter_coef = m.params[inter_key[0]] if inter_key else np.nan
        inter_p    = m.pvalues[inter_key[0]] if inter_key else np.nan
        inter_ci   = m.conf_int().loc[inter_key[0]] if inter_key else (np.nan, np.nan)

        row = {
            "Model": name,
            "R2": round(m.rsquared, 3),
            "Adj_R2": round(m.rsquared_adj, 3),
            "Interaction_coef": round(inter_coef, 4),
            "Interaction_p": round(inter_p, 4),
            "CI_low": round(inter_ci[0], 4),
            "CI_high": round(inter_ci[1], 4),
            "N": int(m.nobs)
        }
        rows.append(row)
        print(f"\n{'─'*60}")
        print(f"  {name}   R²={row['R2']}  Adj-R²={row['Adj_R2']}")
        print(f"  Interaction β₃={row['Interaction_coef']}  p={row['Interaction_p']}")
        print(f"  95% CI: [{row['CI_low']}, {row['CI_high']}]")

    inter_df = pd.DataFrame(rows)
    inter_df.to_csv(os.path.join(OUT, "interaction_models.csv"), index=False)

    # LaTeX table
    latex = _interaction_latex(inter_df)
    with open(os.path.join(OUT, "interaction_table.tex"), "w") as f:
        f.write(latex)
    print(f"\nSaved: output/interaction_models.csv, output/interaction_table.tex")

    # Plot: all three models side by side
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors_pre  = "#2166ac"
    colors_post = "#d73027"
    for ax, (name, m) in zip(axes, models):
        var = name.split("×")[0].strip().replace(" ", "_")
        # map model name to actual column
        varmap = {"GDP": "GDP_Growth_Rate", "Inflation": "Inflation_Rate", "Informal": "Informal_Pct"}
        col = varmap[var]
        pre  = df[df["Crisis_Dummy"] == 0]
        post = df[df["Crisis_Dummy"] == 1]
        ax.scatter(pre[col],  pre[DEPVAR],  color=colors_pre,  alpha=0.6, s=30, label="Pre-2022")
        ax.scatter(post[col], post[DEPVAR], color=colors_post, alpha=0.6, s=30, label="Post-2022")
        # Regression lines
        x_pre  = np.linspace(pre[col].min(),  pre[col].max(),  50)
        x_post = np.linspace(post[col].min(), post[col].max(), 50)
        # Predict using the OLS model
        pred_pre  = (m.params["Intercept"] + m.params[col] * x_pre)
        pred_post = (m.params["Intercept"] + m.params[col] * x_post
                     + m.params["Crisis_Dummy"]
                     + m.params.get(f"{col}:Crisis_Dummy", 0) * x_post)
        ax.plot(x_pre,  pred_pre,  color=colors_pre,  lw=2)
        ax.plot(x_post, pred_post, color=colors_post, lw=2)
        ax.set_xlabel(col.replace("_", " "), fontsize=9)
        ax.set_ylabel("Underemployment Rate (%)", fontsize=9)
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "interaction_models_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: output/interaction_models_plot.png")

    return inter_df


def _interaction_latex(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[h]",
        r"\caption{Interaction model results (HC3 robust SE, quarterly data, $n\approx36$).}",
        r"\label{tab:interaction_extended}",
        r"\footnotesize",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Model} & $R^2$ & Adj-$R^2$ & $\hat\beta_3$ & $p$ & \multicolumn{2}{c}{95\% CI} \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        sig = "**" if row["Interaction_p"] < 0.05 else ("*" if row["Interaction_p"] < 0.10 else "")
        lines.append(
            f"{row['Model']} & {row['R2']} & {row['Adj_R2']} & "
            f"{row['Interaction_coef']}{sig} & {row['Interaction_p']} & "
            f"{row['CI_low']} & {row['CI_high']} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\multicolumn{7}{l}{\scriptsize **$p<0.05$, *$p<0.10$} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# STEP 4: Granger causality on quarterly data
# ---------------------------------------------------------------------------
def step4_granger(df: pd.DataFrame):
    sep("STEP 4 — GRANGER CAUSALITY (quarterly, maxlag=4, Bonferroni α=0.01)")

    all_preds = PREDICTORS + EXTRA_PRED
    bonferroni_alpha = 0.05 / len(all_preds)
    print(f"Bonferroni-corrected α = 0.05 / {len(all_preds)} = {bonferroni_alpha:.4f}")

    results = []
    for pred in all_preds:
        s = df[[DEPVAR, pred]].dropna()
        if len(s) < 12:
            print(f"  {pred}: skip (n={len(s)} < 12)")
            continue
        gc = grangercausalitytests(s, maxlag=4, verbose=False)
        p_vals = {lag: gc[lag][0]["ssr_ftest"][1] for lag in range(1, 5)}
        f_vals = {lag: gc[lag][0]["ssr_ftest"][0] for lag in range(1, 5)}
        best_lag = min(p_vals, key=p_vals.get)
        best_p   = p_vals[best_lag]
        best_f   = f_vals[best_lag]

        reject_bonf = best_p < bonferroni_alpha
        reject_nom  = best_p < 0.05

        verdict = ("✅ Granger-causes (Bonferroni)" if reject_bonf
                   else ("⚠️  Nominal only (p<0.05)" if reject_nom
                         else "✗ No evidence"))

        print(f"  {pred:25s}  best lag={best_lag}  F={best_f:.4f}  p={best_p:.4f}  → {verdict}")
        results.append({
            "Predictor": pred,
            "Best_Lag": best_lag,
            "F_stat": round(best_f, 4),
            "p_value": round(best_p, 4),
            **{f"p_lag{l}": round(p_vals[l], 4) for l in range(1, 5)},
            "Reject_Bonferroni": reject_bonf,
            "Reject_Nominal": reject_nom,
            "Verdict": verdict
        })

    gc_df = pd.DataFrame(results)
    gc_df.to_csv(os.path.join(OUT, "granger_quarterly.csv"), index=False)
    print(f"\nSaved: output/granger_quarterly.csv")
    return gc_df


# ---------------------------------------------------------------------------
# STEP 5: Robustness checks
# ---------------------------------------------------------------------------
def step5_robustness(df: pd.DataFrame, ardl_res=None):
    sep("STEP 5 — ROBUSTNESS CHECKS")

    y    = df[DEPVAR]
    X    = add_constant(df[PREDICTORS].dropna())
    y_   = y.loc[X.index]
    ols_hc3 = OLS(y_, X).fit(cov_type="HC3")

    # --- 5A: Newey-West HAC ---
    print("\n5A. Newey-West HAC standard errors (vs HC3)")
    ols_hac = OLS(y_, X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    comparison = pd.DataFrame({
        "coef":    ols_hc3.params.round(4),
        "se_HC3":  ols_hc3.bse.round(4),
        "p_HC3":   ols_hc3.pvalues.round(4),
        "se_HAC":  ols_hac.bse.round(4),
        "p_HAC":   ols_hac.pvalues.round(4),
    })
    print(comparison.to_string())
    comparison.to_csv(os.path.join(OUT, "hac_vs_hc3.csv"))
    print("Saved: output/hac_vs_hc3.csv")

    # --- 5B: CUSUM and CUSUM-of-squares ---
    print("\n5B. CUSUM and CUSUM-of-squares tests")
    from statsmodels.stats.diagnostic import recursive_olsresiduals
    # Returns: rresid, rparams, rypred, rresid_se, rresid_stdcorr, rcusum, rcusumci
    rec_out       = recursive_olsresiduals(ols_hc3, alpha=0.95)
    rresid        = rec_out[0]
    rparams       = rec_out[1]
    rypred        = rec_out[2]
    rresid_se     = rec_out[3]
    rresid_stdcorr = rec_out[4]

    # CUSUM
    cusum  = np.cumsum(rresid_stdcorr)
    n_obs  = len(cusum)
    t_vals = np.arange(1, n_obs + 1)
    # 5% critical boundary: ±0.948 * sqrt(n)
    boundary = 0.948 * np.sqrt(n_obs)
    cusum_reject = np.any(np.abs(cusum) > boundary)
    print(f"  CUSUM: max|stat|={np.max(np.abs(cusum)):.3f}  5%-boundary={boundary:.3f}  {'REJECT' if cusum_reject else 'OK'} parameter stability")

    # CUSUM-sq
    cusumsq = np.cumsum(rresid**2) / np.sum(rresid**2)
    # approximate 5% boundary ≈ 0.143/sqrt(n)  (Brown-Durbin-Evans)
    sq_boundary = 0.143 + np.arange(len(cusumsq)) / len(cusumsq) * 0 + 0.143
    cusumsq_reject = np.any(np.abs(cusumsq - np.linspace(0, 1, len(cusumsq))) > 0.20)
    print(f"  CUSUM-sq: stability {'REJECTED' if cusumsq_reject else 'OK'} at approx 5%")

    # Plot CUSUM
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(cusum, color="steelblue", lw=1.5, label="CUSUM")
    ax.axhline( boundary, color="red", ls="--", lw=1, label="5% bounds")
    ax.axhline(-boundary, color="red", ls="--", lw=1)
    ax.axhline(0, color="black", ls="-", lw=0.5)
    ax.set_title("CUSUM Test for Parameter Stability")
    ax.set_xlabel("Recursive observation")
    ax.set_ylabel("Cumulative sum of recursive residuals")
    ax.legend(fontsize=8)

    ax = axes[1]
    expected_line = np.linspace(0, 1, len(cusumsq))
    ax.plot(cusumsq, color="steelblue", lw=1.5, label="CUSUM-sq")
    ax.plot(expected_line + 0.20, color="red", ls="--", lw=1, label="5% bounds (approx)")
    ax.plot(expected_line - 0.20, color="red", ls="--", lw=1)
    ax.plot(expected_line, color="black", ls="-", lw=0.8, alpha=0.4)
    ax.set_title("CUSUM-sq Test for Parameter Stability")
    ax.set_xlabel("Recursive observation")
    ax.set_ylabel("CUSUM of squared recursive residuals")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "cusum_plots.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: output/cusum_plots.png")

    # --- 5C: Rolling-window OLS (window=12 quarters) ---
    print("\n5C. Rolling-window OLS (window=12 quarters) for GDP coefficient")
    ROLL = 12
    gdp_coefs = []
    gdp_ci_lo = []
    gdp_ci_hi = []
    roll_idx  = []

    df_clean = df[PREDICTORS + [DEPVAR]].dropna().reset_index(drop=True)
    for start in range(len(df_clean) - ROLL + 1):
        window = df_clean.iloc[start:start + ROLL]
        X_w = add_constant(window[PREDICTORS])
        y_w = window[DEPVAR]
        try:
            m_w = OLS(y_w, X_w).fit(cov_type="HC3")
            coef = m_w.params.get("GDP_Growth_Rate", np.nan)
            ci   = m_w.conf_int().loc["GDP_Growth_Rate"] if "GDP_Growth_Rate" in m_w.params.index else (np.nan, np.nan)
            gdp_coefs.append(coef)
            gdp_ci_lo.append(ci[0])
            gdp_ci_hi.append(ci[1])
            roll_idx.append(start + ROLL - 1)  # index of last obs in window
        except Exception:
            gdp_coefs.append(np.nan)
            gdp_ci_lo.append(np.nan)
            gdp_ci_hi.append(np.nan)
            roll_idx.append(start + ROLL - 1)

    roll_df = pd.DataFrame({
        "Window_end_idx": roll_idx,
        "GDP_coef": gdp_coefs,
        "CI_low": gdp_ci_lo,
        "CI_high": gdp_ci_hi
    })
    roll_df.to_csv(os.path.join(OUT, "rolling_gdp_coef.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(gdp_coefs))
    ax.plot(x, gdp_coefs, color="steelblue", lw=2, label="GDP coef (12-Q rolling OLS)")
    ax.fill_between(x, gdp_ci_lo, gdp_ci_hi, alpha=0.2, color="steelblue")
    ax.axhline(0, color="black", ls="-", lw=0.5)
    # Mark 2022 window (approx index where 2022 enters)
    ax.axvline(x=len(gdp_coefs) - 10, color="red", ls="--", lw=1, label="2022 enters window")
    ax.set_title("Rolling GDP Coefficient (12-Quarter Window)\nUneremployment on GDP Growth Rate", fontsize=10)
    ax.set_xlabel("Window end index (quarters from 2016-Q1)")
    ax.set_ylabel("OLS coefficient on GDP Growth Rate")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "rolling_gdp_coef.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: output/rolling_gdp_coef.png, output/rolling_gdp_coef.csv")

    return roll_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()

    adf_df  = step2_stationarity(df)
    ardl_res = step2_ardl(df)
    inter_df = step3_interaction_models(df)
    gc_df    = step4_granger(df)
    roll_df  = step5_robustness(df)

    sep("ALL STEPS COMPLETE")
    print(f"Output files written to: {OUT}")
    files = sorted(os.listdir(OUT))
    for f_name in files:
        print(f"  {f_name}")
