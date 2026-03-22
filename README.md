# Analysing the Economic Drivers of Underemployment in Sri Lanka: A Data-Driven Investigation of the 2015–2025 Period

**Authors**: Ekanayake T.N.D.S.W., Bulagala D.W.K.G., De Silva B.K.P., Pabasara W.G.K., Lelwala J.U.P.  
**Institution**: Department of Computer Science & Engineering, University of Moratuwa, Moratuwa, Sri Lanka  

---

## Abstract
Underemployment in Sri Lanka — encompassing both time-related underemployment (employed persons working fewer than 40 hours per week while willing and available to work more) and skills-based underemployment (workers engaged in roles below their qualification level) — represents a growing and underanalysed dimension of labour market slack. Time-related underemployment alone rose sharply from 4.6% in 2019 to 8.3% during the 2022 economic crisis, yet both forms remain absent from the country’s existing labour market literature, which focuses exclusively on the aggregate unemployment rate. This study investigates the economic drivers of underemployment in Sri Lanka through a systematic data analysis of quarterly Labour Force Survey (LFS) data spanning 2015 Q1 to 2025 Q3. Using exploratory data analysis, correlation analysis, structural break detection, Granger causality tests, and ARDL/VECM long-run modelling — supplemented by SHAP-based feature association analysis — the study answers four targeted research questions about which macroeconomic indicators are most strongly associated with underemployment, how the 2022 economic crisis altered these relationships, and whether persistent structural patterns can be identified across the recovery period. The findings are translated into evidence-based policy recommendations for Sri Lanka’s labour market stakeholders.

## 1. Problem Motivation
Sri Lanka’s labour market conceals significant underutilisation beneath its headline unemployment rate of approximately 3.8%. Underemployment takes two interrelated forms in this context:
* **Time-related underemployment**: employed persons constrained to fewer than 40 working hours per week who desire additional work — this rate increased by approximately 80% between 2019 and 2022, rising from 4.6% to 8.3%.
* **Skills-based (qualification) underemployment**: workers engaged in roles that do not utilise their educational attainment or professional competencies — a structurally persistent phenomenon driven by graduate labour market mismatches and public sector hiring freezes.

Both dimensions surged during the most severe economic crisis in Sri Lanka’s post-independence history: sovereign default, foreign exchange reserve collapse, inflation exceeding 70%, tourism sector contraction, and acute supply chain disruptions. Despite subsequent IMF stabilisation under the Extended Fund Facility programme, preliminary indicators suggest underemployment has yet to return to pre-crisis levels, pointing to persistent structural labour market slack.

This problem warrants investigation for three interconnected reasons:
1. Hours-based underemployment — not headline unemployment — is the primary driver of suppressed wage growth in post-recession economies, making it the most policy-relevant indicator of labour market health during Sri Lanka’s recovery.
2. Existing Sri Lanka labour economics studies that use time-series data focus on the aggregate unemployment rate and end their data periods at 2020 Q4 or earlier, entirely omitting the 2022 crisis and leaving the LFS quarterly underemployment series unanalysed.
3. No existing study conducts a data-driven analysis of the economic factors associated with underemployment dynamics in Sri Lanka across the crisis and recovery periods.

## 2. Research Questions
This study is organised around four research questions answered through data analysis:
* **RQ1**. How did underemployment patterns in Sri Lanka evolve across the 2015–2025 period, and what distinguishes the crisis phase (2022–2023) from the preceding and recovery periods?
* **RQ2**. Which macroeconomic indicators — among GDP growth, youth labour force participation rate (LFPR), services sector employment share, agricultural output, and remittance inflows — show the strongest statistical association with underemployment?
* **RQ3**. Did the 2022 economic crisis structurally alter the relationship between macroeconomic indicators and underemployment, and if so, how?
* **RQ4**. Is there evidence of a long-run equilibrium relationship between key macroeconomic indicators and the underemployment rate?

## 3. Identified Data Sources
The study constructs a quarterly macro-financial dataset spanning 2015 Q1 to 2025 Q3 (n=43 observations), combining official national and international portals.

**Primary Labour Market Data:**
* Department of Census and Statistics (DCS) Sri Lanka — Quarterly Labour Force Survey (LFS) micro-data, 2015–2025: the source of the primary variable of interest (underemployment rate, %) — covering both hours-based and qualification-based dimensions — and youth Labour Force Participation Rate (LFPR, %).

**Macroeconomic and Sectoral Indicators:**
* Central Bank of Sri Lanka (CBSL) — Quarterly real GDP growth rate (%), services sector employment share (%), remittance inflows (USD millions), and crisis period classification.
* World Bank DataBank — Cross-validated real GDP growth series (seasonally adjusted).
* ILOSTAT / ILO Database — Youth LFPR (ages 15–24, %), cross-checked against DCS LFS estimates.
* Food and Agriculture Organization (FAO) & CEIC Data — Quarterly agricultural output index (base year normalised; paddy, vegetable, and tea sub-indices available).
* Sri Lanka Meteorological Department — Quarterly monsoon indicators and agricultural seasonal index, used as control variables.

## 4. Proposed Methodology
### Variable Definition
The primary variable of interest is the composite underemployment rate derived from LFS quarterly data. Where disaggregation is feasible, two sub-series will be analysed: (i) the time-related underemployment rate (% employed working <40 hrs/week wanting more hours) and (ii) a qualification-based proxy constructed from occupational mismatch indicators in LFS micro-data. A sensitivity analysis will confirm whether findings are consistent across both definitions.

### Phase 1 — Data Preparation
Raw series will be validated for consistency across DCS, CBSL, and international sources. Missing values will be addressed using KNN imputation for isolated gaps and MICE for blocks of missing observations during 2019–2022.

### Phase 2 — Exploratory Data Analysis (EDA)
EDA will characterise the temporal behaviour of underemployment and each macroeconomic indicator across 2015–2025, directly addressing RQ1. This includes time-series plots to identify co-movement patterns; period-segmented descriptive statistics across three phases (pre-crisis: 2015–2019, crisis: 2020–2023, recovery: 2024–2025); correlation heatmaps and scatter plots disaggregated by crisis and non-crisis periods; and STL seasonal decomposition to isolate trend, seasonal, and residual components of the underemployment series.

### Phase 3 — Statistical Association Analysis
Formal statistical tests will quantify the relationships identified in EDA, addressing RQ2 and RQ3:
* **Stationarity testing**: Augmented Dickey-Fuller (ADF) and KPSS tests will assess integration order for each series, a prerequisite for valid time-series inference.
* **Structural break detection**: Zivot-Andrews and Bai-Perron tests will determine whether the 2022 crisis constitutes a statistically significant break in the underemployment series and in its relationships with each indicator.
* **Correlation and crisis interaction analysis**: Pearson and Spearman correlations, computed separately for pre-crisis and post-crisis subsamples, will reveal shifts in association strength and direction. Regression models with interaction terms (e.g., GDP growth × crisis dummy) will quantify how the crisis moderated each indicator’s relationship with underemployment.
* **Multicollinearity assessment**: Variance Inflation Factors (VIF) will be computed across all five indicators; PCA applied as a fallback if VIF > 10 for any variable pair.

### Phase 4 — Long-Run Relationship Analysis
To address RQ4, the study tests for and estimates long-run equilibrium relationships. Granger causality tests ($\alpha = 0.05$, Bonferroni-corrected for five predictors) will assess whether lagged indicator values carry statistically significant information about underemployment. Johansen cointegration tests will then determine whether a long-run equilibrium exists between underemployment and the indicator set. Conditional on these results, an ARDL model or VECM will be estimated to produce interpretable long-run coefficients quantifying the direction and magnitude of each indicator’s association with underemployment.

### Phase 5 — SHAP-Based Feature Association Analysis
To complement the econometric analysis, SHAP will be applied to an XGBoost model fitted on the full dataset. SHAP values will be used strictly as a measure of statistical association strength — not as causal or predictive claims — yielding a data-driven ranking of indicators by their contribution to explaining variation in underemployment. This ranking will be cross-validated against Granger and ARDL/VECM results, with convergence or divergence across methods explicitly discussed. Force plots and waterfall charts will additionally be generated for the 2022 crisis peak quarter.

---

## Repository Structure & Key Deliverables
This repository contains data extraction pipelines and comprehensive analysis implementing the methodology described above. 

1. **[Final Underemployment Dashboard](Data_Analysis/Final_Underemployment_Dashboard.ipynb)**: Start here. This master presentation dashboard displays the quarterly underemployment ACF/PACF structural bounds, global correlation matrices against economic drivers, and multi-axis macro trend overlays.
2. **Individual Indicator Deep Dives (`/Data_Analysis`)**: Extensive notebooks developed by the team to analyze specific linkages:
   - `Agriculture_Output_vs_Underemployment.ipynb`
   - `GDP_Growth_vs_Underemployment_Lags.ipynb`
   - `Remittances_vs_Underemployment.ipynb`
   - `Youth_LFPR_vs_Underemployment.ipynb`
3. **Data Extraction Pipelines (`/extraction`)**: Custom Python parsing logic used to convert localized quarterly LFS microdata into an accurate, weighted quarterly target variable baseline (`quarterly_underemployment.csv`) and construct the qualification-based proxy.
4. **Machine Learning Explanations (`run_shap_analysis.py`, `lagged_analysis.py`)**: XGBoost algorithms natively implemented to dissect predictive importance using SHAP techniques.
5. **Advanced Diagnostic Visualizations (`Data_Analysis/Advanced_EDA_Dashboard.ipynb`)**: Deeper exploratory data analysis characterizing the temporal behavior of underemployment.
6. **Structural Break & Interaction Coefficients (RQ3)**: Addressed in `Data_Analysis/RQ3_Interaction_Report.ipynb`, finding structural breaks via Zivot-Andrews testing and quantifying elasticity shifts with crisis interaction models.

## Setup Requirements
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt # containing pandas, seaborn, scikit-learn, shap, statsmodels, xgboost
```
