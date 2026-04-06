# Data Dictionary & Existing Data Audit

**Date of Audit**: April 2026 (Initial review of existing `master_dataset.csv`)
**Auditor**: Member 3 — Bulagala D.W.K.G. (Data Architect)

## 1. Overview of Current Dataset
- **File Checked**: `DataLoader/master_dataset.csv`
- **Total Variables Listed**: 92 (including granular agricultural sub-indices and demographic splits)
- **Timeframe**: 2015 to 2024 (Annual dataset)
- **Missing Data Warning**: Several values in 2024 (such as Real GDP and GDP Growth Rate) are currently blank strings and will need to be properly typed as `NaN` before MICE/KNN imputation.

## 2. Core 10 Macroeconomic Variables

Below is the initial audit of the 10 critical variables identified for the project, noting their sources as per the `README.md`:

| Variable Name | Exact Source | Access Date | Prior Imputation/Adjustments Noted |
|---|---|---|---|
| **Underemployment Rate (%)** | DCS Sri Lanka - Qtly Labour Force Survey | *To be recorded* | **WARNING**: 2022 value exhibits LFS Artefact. Currently reported as ~2.3% but known to be heavily distorted by the crisis. Needs adjustment/imputation. |
| **Youth LFPR (15-24) (%)** | ILOSTAT & DCS LFS | *To be recorded* | None noted yet. |
| **Real GDP Growth Rate (%)** | CBSL / World Bank DataBank | *To be recorded* | Missing values observed for 2024 in current dataset. Needs updating. |
| **Services Sector Emp. Share (%)** | CBSL | *To be recorded* | Needs verification in `master_dataset.csv` (currently masked under informal/industry subsets). |
| **Remittance Inflows (USD)** | World Bank DataBank | *To be recorded* | Present as `Remit_Personal_remittances_...` |
| **Agricultural Output Index** | FAOSTAT & CEIC | *To be recorded* | Currently represented by 60+ granular crop columns (e.g., `AgriProdIdx_Tea_leaves`, `AgriProdIdx_Paddy`). **ACTION**: Must compress into a single weighted index per our plan. |
| **Nominal Exchange Rate (LKR/USD)** | CBSL (2015-2020) & FRED (2021-2024) | *To be recorded* | **ACTION**: Must calculate the *Real* Exchange rate using CPI deflators. Currently only a raw numerical rate is saved. |
| **Inflation Rate (%)** | CBSL (CCPI) / FRED | *To be recorded* | Present as `Inflation_Rate`. |
| **Informal Employment (%)** | DCS Sri Lanka | *To be recorded* | Present as `Informal_Pct`. |
| **Unemployment Rate (%)** | DCS Sri Lanka | *To be recorded* | Present as `Unemployment_Rate`. |

## 3. Notable Issues Discovered in Audit

1. **The 2022 LFS Underemployment Artefact**: The dataset currently uses the flawed 2.3% annual average. We must replace this in the annual file with an imputed `(2021 + 2023) / 2` measure and flag it.
2. **Missing 2024 Values**: Several rows in 2024 are incomplete (e.g. Real GDP). These will be targeted by our MICE/KNN imputation scrips.
3. **Agricultural Bloat**: The dataset contains 60+ columns for individual crops (cabbages, carrots, avocados). These must be aggregated using FAOSTAT production values into a single `AgriProdIdx_Weighted` variable.

## 4. Next Steps
Move to **Step 2: Reconstruct the real exchange rate series**. We will write the `03_exchange_rate.py` script to fetch CPI and splice the FRED/CBSL datasets.
