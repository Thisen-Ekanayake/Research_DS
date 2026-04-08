"""
build_quarterly_dataset.py
==========================
Step 1 of the Member 1 (Ekanayake) econometric pipeline.

Assembles the quarterly master dataset (n≈40, 2015 Q1 – 2024 Q4) by:
  1. Loading quarterly series that already exist (underemployment, exchange rate,
     remittances, agricultural output index).
  2. Applying Denton-Cholette temporal disaggregation (flat indicator = proportional
     Denton) to annual-only series: Real_GDP levels → quarterly GDP growth;
     CPI levels → quarterly inflation; Informal_Pct; Youth_LFPR_15_24.
  3. Merging everything on (Year, Quarter) index.
  4. Adding crisis dummy (1 from 2022-Q1 onward).
  5. Verifying stationarity readiness and exporting.

Output: ardl_vecm/quarterly_master_dataset.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from statsmodels.tsa.interp.denton import dentonm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)

ANNUAL_CSV   = os.path.join(BASE, "master_dataset_2025.csv")
Q_UNEMP      = os.path.join(ROOT, "labour", "finalized_csv", "quarterly_underemployment.csv")
Q_FX         = os.path.join(ROOT, "data_pipeline", "quarterly_exchange_rates.csv")
Q_REMIT      = os.path.join(ROOT, "data_pipeline", "output", "quarterly_remittances.csv")
Q_AGRI       = os.path.join(ROOT, "data_pipeline", "output", "quarterly_agricultural_output.csv")
OUT_CSV      = os.path.join(BASE, "quarterly_master_dataset.csv")

STUDY_YEARS  = list(range(2015, 2025))   # 2015–2024 inclusive


# ---------------------------------------------------------------------------
# Helper: Denton proportional disaggregation with flat indicator
# ---------------------------------------------------------------------------
def denton_flat(annual_series: np.ndarray) -> np.ndarray:
    """
    Disaggregate an annual series to quarterly using Denton-Cholette with a
    flat (constant = 1.0) indicator.  With a flat indicator, dentonm minimises
    the sum of squared first differences of the quarterly series subject to the
    annual sum constraint → smooth interpolation.

    Parameters
    ----------
    annual_series : 1-D array of annual values (n_years,)

    Returns
    -------
    quarterly : 1-D array of length n_years * 4
    """
    n_years = len(annual_series)
    # indicator is the HIGH-frequency (quarterly) flat series
    indicator = np.ones(n_years * 4)
    # benchmark is the LOW-frequency (annual) series
    quarterly = dentonm(indicator, annual_series, freq="aq")
    return quarterly


def annual_to_quarterly_df(annual_values: np.ndarray,
                            years: list,
                            col_name: str,
                            is_rate: bool = False) -> pd.DataFrame:
    """
    Wrap denton_flat output into a (Year, Quarter, col_name) DataFrame.

    Parameters
    ----------
    is_rate : bool
        If True, the annual value is a RATE (annual = average of quarterly).
        Multiply benchmark by 4 so each quarter ≈ the annual rate.
        If False, the annual value is a FLOW/LEVEL (annual = sum of quarterly).
    """
    if is_rate:
        # sum(Q1..Q4) = annual * 4  →  mean(Q1..Q4) = annual
        q_values = denton_flat(annual_values * 4)
    else:
        q_values = denton_flat(annual_values)

    quarters = ["Q1", "Q2", "Q3", "Q4"]
    rows = []
    for i, yr in enumerate(years):
        for j, qtr in enumerate(quarters):
            rows.append({"Year": yr, "Quarter": qtr, col_name: q_values[i * 4 + j]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Load quarterly underemployment
# ---------------------------------------------------------------------------
def load_quarterly_underemployment() -> pd.DataFrame:
    df = pd.read_csv(Q_UNEMP)
    df.columns = df.columns.str.strip()
    # Columns: YEAR, QUARTER, total_employed_weighted, underemployed_weighted, underemp_rate, sample_size
    df = df.rename(columns={
        "YEAR": "Year",
        "QUARTER": "Quarter",
        "underemp_rate": "Underemployment_Rate",
        "sample_size": "Sample_Size"
    })
    df["Quarter"] = df["Quarter"].astype(str).str.strip()
    df["Quarter"] = df["Quarter"].apply(lambda x: f"Q{x}" if x.isdigit() else x)
    df = df[df["Year"].isin(STUDY_YEARS)][["Year", "Quarter", "Underemployment_Rate", "Sample_Size"]]

    # Drop clearly erroneous rows (n<100 or rate > 90%)
    n_before = len(df)
    df = df[~((df["Sample_Size"] < 100) | (df["Underemployment_Rate"] > 90))]
    if len(df) < n_before:
        print(f"[unemp]  Dropped {n_before - len(df)} bad rows (tiny sample or rate > 90%)")

    # Build a full (Year, Quarter) spine and fill missing with NaN → interpolate
    spine_rows = [{"Year": yr, "Quarter": f"Q{q}"}
                  for yr in STUDY_YEARS for q in range(1, 5)]
    spine = pd.DataFrame(spine_rows)
    df = spine.merge(df[["Year", "Quarter", "Underemployment_Rate"]], on=["Year", "Quarter"], how="left")

    n_missing = df["Underemployment_Rate"].isna().sum()
    if n_missing > 0:
        df["Underemployment_Rate"] = df["Underemployment_Rate"].interpolate(method="linear")
        print(f"[unemp]  Interpolated {n_missing} missing quarters (linear)")

    print(f"[unemp]  rows={len(df)}, years={sorted(df['Year'].unique())}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Load quarterly exchange rate
# ---------------------------------------------------------------------------
def load_quarterly_fx() -> pd.DataFrame:
    df = pd.read_csv(Q_FX, index_col=0, parse_dates=True)
    df.index.name = "date"
    df.columns = df.columns.str.strip()
    # Column: nominal_lkr_usd
    df = df.rename(columns={"nominal_lkr_usd": "Exchange_Rate"})
    df = df[["Exchange_Rate"]].dropna()

    # Convert date index → (Year, Quarter)
    rows = []
    for date, row in df.iterrows():
        yr = date.year
        month = date.month
        qtr_num = (month - 1) // 3 + 1
        rows.append({"Year": yr, "Quarter": f"Q{qtr_num}",
                     "Exchange_Rate": row["Exchange_Rate"]})
    out = pd.DataFrame(rows)
    out = out[out["Year"].isin(STUDY_YEARS)]
    print(f"[fx]     rows={len(out)}, years={sorted(out['Year'].unique())}")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Load quarterly remittances (Denton-disaggregated WB benchmark)
# ---------------------------------------------------------------------------
def load_quarterly_remittances() -> pd.DataFrame:
    df = pd.read_csv(Q_REMIT)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"Disaggregated_WB_Remittance": "Remittances_USD"})
    df["Quarter"] = df["Quarter"].astype(str).str.strip()
    df = df[df["Year"].isin(STUDY_YEARS)][["Year", "Quarter", "Remittances_USD"]]
    print(f"[remit]  rows={len(df)}, years={sorted(df['Year'].unique())}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Load quarterly agricultural output index
# ---------------------------------------------------------------------------
def load_quarterly_agri() -> pd.DataFrame:
    df = pd.read_csv(Q_AGRI)
    df.columns = df.columns.str.strip()
    # Columns: Year, Quarter, Indicator, FAO_Index_Quarterly
    df = df.rename(columns={"FAO_Index_Quarterly": "Agri_Output_Index"})
    df["Quarter"] = df["Quarter"].astype(str).str.strip()
    df = df[df["Year"].isin(STUDY_YEARS)][["Year", "Quarter", "Agri_Output_Index"]]
    print(f"[agri]   rows={len(df)}, years={sorted(df['Year'].unique())}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Denton-disaggregate annual-only series from master_dataset_2025.csv
# ---------------------------------------------------------------------------
def load_and_disaggregate_annual() -> pd.DataFrame:
    ann = pd.read_csv(ANNUAL_CSV)
    ann.columns = ann.columns.str.strip()
    # Strip whitespace from all object columns and coerce numeric columns
    for col in ann.columns:
        if ann[col].dtype == object:
            ann[col] = ann[col].str.strip()
    ann = ann.replace('', np.nan)
    # Coerce numeric columns
    for col in ["Real_GDP", "GDP_Growth_Rate", "Inflation_Rate", "Informal_Pct", "Youth_LFPR_15_24"]:
        ann[col] = pd.to_numeric(ann[col], errors='coerce')

    ann = ann[ann["Year"].isin(STUDY_YEARS)].sort_values("Year").reset_index(drop=True)

    # Impute 2024 missing GDP values (World Bank IMF estimate: ~5.0% growth)
    idx_2024 = ann.index[ann["Year"] == 2024]
    if len(idx_2024) > 0:
        i24 = idx_2024[0]
        if pd.isna(ann.at[i24, "GDP_Growth_Rate"]):
            ann.at[i24, "GDP_Growth_Rate"] = 5.0
            print("[annual] Imputed 2024 GDP_Growth_Rate = 5.0% (World Bank/IMF estimate)")
        if pd.isna(ann.at[i24, "Real_GDP"]):
            idx_2023 = ann.index[ann["Year"] == 2023]
            if len(idx_2023) > 0:
                gdp_2023 = ann.at[idx_2023[0], "Real_GDP"]
                ann.at[i24, "Real_GDP"] = gdp_2023 * (1 + ann.at[i24, "GDP_Growth_Rate"] / 100)
                print(f"[annual] Imputed 2024 Real_GDP = {ann.at[i24, 'Real_GDP']:.2f}")

    years = list(ann["Year"].values)
    n = len(years)
    assert n == len(STUDY_YEARS), f"Expected {len(STUDY_YEARS)} annual rows, got {n}"

    results = {}

    # --- Real GDP levels → quarterly levels → quarterly YoY growth rate ---
    gdp_levels = ann["Real_GDP"].values.astype(float)
    gdp_q_df = annual_to_quarterly_df(gdp_levels, years, "Real_GDP_Q")

    # Compute quarterly GDP growth as year-on-year % change
    # Arrange in order then do pct_change vs 4 periods ago
    gdp_q_df = gdp_q_df.sort_values(["Year", "Quarter"]).reset_index(drop=True)
    gdp_q_df["GDP_Growth_Rate"] = gdp_q_df["Real_GDP_Q"].pct_change(periods=4) * 100
    # Drop the first year (NaN due to lag-4 pct_change)
    gdp_q_df = gdp_q_df.drop(columns=["Real_GDP_Q"])
    results["gdp"] = gdp_q_df

    # --- CPI levels → quarterly CPI levels → quarterly YoY inflation ---
    # Reconstruct CPI index from annual inflation rates:
    # CPI_2015 = 100.0 (base), CPI_{t} = CPI_{t-1} * (1 + infl_t/100)
    infl = ann["Inflation_Rate"].values.astype(float)
    cpi_levels = np.zeros(n)
    cpi_levels[0] = 100.0
    for i in range(1, n):
        cpi_levels[i] = cpi_levels[i - 1] * (1 + infl[i] / 100.0)

    cpi_q_df = annual_to_quarterly_df(cpi_levels, years, "CPI_Q")
    cpi_q_df = cpi_q_df.sort_values(["Year", "Quarter"]).reset_index(drop=True)
    # Quarterly YoY inflation = (CPI_t - CPI_{t-4}) / CPI_{t-4} * 100
    cpi_q_df["Inflation_Rate"] = cpi_q_df["CPI_Q"].pct_change(periods=4) * 100
    cpi_q_df = cpi_q_df.drop(columns=["CPI_Q"])
    results["cpi"] = cpi_q_df

    # --- Informal Employment % (rate: annual = average of quarterly) ---
    informal = ann["Informal_Pct"].values.astype(float)
    inf_q_df = annual_to_quarterly_df(informal, years, "Informal_Pct", is_rate=True)
    results["informal"] = inf_q_df

    # --- Youth LFPR (15–24) (rate: annual = average of quarterly) ---
    youth = ann["Youth_LFPR_15_24"].values.astype(float)
    youth_q_df = annual_to_quarterly_df(youth, years, "Youth_LFPR", is_rate=True)
    results["youth"] = youth_q_df

    # Merge all annual-disaggregated series
    base = results["gdp"]
    for key in ["cpi", "informal", "youth"]:
        base = base.merge(results[key], on=["Year", "Quarter"], how="left")

    print(f"[annual] disaggregated rows={len(base)}, cols={list(base.columns)}")
    return base


# ---------------------------------------------------------------------------
# 6. Merge everything
# ---------------------------------------------------------------------------
def build_quarterly_dataset() -> pd.DataFrame:
    unemp  = load_quarterly_underemployment()
    fx     = load_quarterly_fx()
    remit  = load_quarterly_remittances()
    agri   = load_quarterly_agri()
    annual = load_and_disaggregate_annual()

    # Start from underemployment as the spine
    df = unemp.copy()
    for frame in [fx, remit, agri, annual]:
        df = df.merge(frame, on=["Year", "Quarter"], how="left")

    # Sort by time
    q_order = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
    df["q_num"] = df["Quarter"].map(q_order)
    df = df.sort_values(["Year", "q_num"]).drop(columns=["q_num"]).reset_index(drop=True)

    # Add crisis dummy: 1 from 2022-Q1 onwards (sovereign default)
    df["Crisis_Dummy"] = ((df["Year"] > 2022) |
                          ((df["Year"] == 2022) & (df["Quarter"] >= "Q1"))).astype(int)

    # Create a proper PeriodIndex label for readability
    df["Period"] = df["Year"].astype(str) + "-" + df["Quarter"]

    # Re-order columns cleanly
    col_order = [
        "Period", "Year", "Quarter",
        "Underemployment_Rate",
        "GDP_Growth_Rate",
        "Inflation_Rate",
        "Exchange_Rate",
        "Informal_Pct",
        "Youth_LFPR",
        "Remittances_USD",
        "Agri_Output_Index",
        "Crisis_Dummy",
    ]
    df = df[col_order]

    return df


# ---------------------------------------------------------------------------
# 7. Validate and export
# ---------------------------------------------------------------------------
def validate(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("QUARTERLY DATASET VALIDATION")
    print("=" * 60)
    print(f"Shape         : {df.shape}")
    print(f"Period range  : {df['Period'].iloc[0]} → {df['Period'].iloc[-1]}")
    print(f"NaN counts    :\n{df.isnull().sum().to_string()}")

    # Check GDP growth YoY NaNs are only in first 4 rows
    gdp_nans = df[df["GDP_Growth_Rate"].isna()]
    if len(gdp_nans) > 4:
        print(f"WARNING: {len(gdp_nans)} NaN rows in GDP_Growth_Rate — expected 4 (first year)")
    else:
        print(f"GDP_Growth_Rate: {len(gdp_nans)} NaN rows in first year (expected, YoY lag)")

    infl_nans = df[df["Inflation_Rate"].isna()]
    if len(infl_nans) > 4:
        print(f"WARNING: {len(infl_nans)} NaN rows in Inflation_Rate — expected 4")
    else:
        print(f"Inflation_Rate : {len(infl_nans)} NaN rows in first year (expected, YoY lag)")

    print("\nFirst 8 rows:")
    print(df.head(8).to_string(index=False))
    print("\nLast 4 rows:")
    print(df.tail(4).to_string(index=False))

    # Basic sanity: underemployment quarterly peaks can reach ~33% (crisis 2020 Q2)
    bad_unemp = df[~df["Underemployment_Rate"].between(1, 40)]
    assert len(bad_unemp) == 0, f"Underemployment_Rate out of [1,40]:\n{bad_unemp}"
    # Exchange rate should be in LKR/USD range (100–400)
    assert df["Exchange_Rate"].between(100, 400).all(), "Exchange_Rate out of range"
    # Informal Pct should be near annual values (~56–61%)
    assert df["Informal_Pct"].between(50, 70).all(), \
        f"Informal_Pct out of range: min={df['Informal_Pct'].min():.1f} max={df['Informal_Pct'].max():.1f}"
    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    print("Building quarterly master dataset...")
    df = build_quarterly_dataset()
    validate(df)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nExported: {OUT_CSV}")
    print(f"Final shape: {df.shape}  (rows, cols)")
