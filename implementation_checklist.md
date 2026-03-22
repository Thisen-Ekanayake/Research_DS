# Implementation Checklist
**Analysing the Economic Drivers of Underemployment in Sri Lanka**

---

## Data Collection & Preparation
- [x] Collect quarterly LFS micro-data from DCS (2015 Q1 – 2025 Q3)
- [x] Pull GDP growth, services sector share, remittances from CBSL
- [x] Cross-validate GDP with World Bank DataBank
- [x] Get youth LFPR from ILOSTAT, cross-check with DCS
- [x] Get agricultural output index from FAO/CEIC
- [ ] Get monsoon/seasonal indicators from Met Department
- [x] Handle missing values: KNN imputation (isolated), MICE (blocks)
- [x] Construct qualification-based underemployment proxy from occupational mismatch in LFS micro-data
- [x] Add crisis dummy (2022) and seasonal quarter dummies

---

## Phase 1 – EDA
- [x] Time-series plots for all variables
- [x] Period-segmented descriptive stats (pre-crisis / crisis / recovery)
- [x] Correlation heatmaps split by crisis vs non-crisis periods
- [x] STL decomposition on underemployment series

---

## Phase 2 – Statistical Analysis
- [x] ADF + KPSS stationarity tests
- [x] Zivot-Andrews + Bai-Perron structural break tests
- [x] Pearson & Spearman correlations (subsamples)
- [x] Regression with interaction terms (e.g. GDP × crisis dummy)
- [x] VIF check; PCA fallback if VIF > 10

---

## Phase 3 – Long-Run Modelling
- [x] Granger causality tests (α=0.05, Bonferroni-corrected)
- [x] Johansen cointegration test
- [x] ARDL or VECM estimation (conditional on cointegration result)

---

## Phase 4 – SHAP Analysis
- [x] Fit XGBoost on full dataset
- [x] Generate SHAP summary, force, and dependence plots
- [x] Focus force/waterfall plots on 2022 crisis peak quarter
- [x] Cross-validate SHAP rankings against Granger/ARDL results

---

## Outputs
- [x] Descriptive report of underemployment dynamics (RQ1)
- [x] Ranked indicator list by association strength (RQ2)
- [x] Structural break report with interaction coefficients (RQ3)
- [x] ARDL/VECM long-run coefficients with CIs (RQ4)
- [x] Policy brief mapping top indicators to interventions
- [x] Description of outputs and methods
- [x] Sensitivity analysis across both underemployment definitions
