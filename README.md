# Forecasting Underemployment in Sri Lanka: Macro-Economic Drivers

This repository contains data extraction pipelines and comprehensive analysis on the macroeconomic drivers affecting the **Quarterly Underemployment Rate** in Sri Lanka. It specifically bridges the gap between raw quarterly Labour Force Survey (LFS) microdata (2015–2024) and macroeconomic time-series from the World Bank and FRED.

The project aligns with the research proposal goal of using Econometric (SARIMA, ARDL / VECM) and Machine Learning (XGBoost, LSTM) models to forecast both time-related and skills-based underemployment.

## Key Deliverables
1. **[Final Underemployment Dashboard](Data_Analysis/Final_Underemployment_Dashboard.ipynb)**: Start here. This master presentation dashboard displays the quarterly underemployment ACF/PACF structural bounds, global correlation matrices against economic drivers, and multi-axis macro trend overlays.
2. **Individual Indicator Deep Dives (`/Data_Analysis`)**: Extensive notebooks developed by the team to analyze specific linkages:
   - `Agriculture_Output_vs_Underemployment.ipynb`
   - `GDP_Growth_vs_Underemployment_Lags.ipynb`
   - `Remittances_vs_Underemployment.ipynb`
   - `Youth_LFPR_vs_Underemployment.ipynb`
3. **Data Extraction Pipelines (`/extraction`)**: Custom Python parsing logic used to convert over 5GB of localized quarterly LFS microdata spanning 10 years into an accurate, weighted quarterly target variable baseline (`quarterly_underemployment.csv`).
4. **Machine Learning Explanations**: Early baseline Gradient Boosting and Random Forest algorithms natively implemented (`run_shap_analysis.py`, `lagged_analysis.py`) to dissect predictive importance using SHAP techniques.

## Findings Summarized 
- **Time Lags & Autocorrelation**: Quarter-to-quarter structural dependencies exist strongly, paving the way for the SARIMA baseline models.
- **Economic Shocks**: Major drops in Real GDP and high spikes in Inflation closely shadow periods where underemployment surges as workers downgrade to lower working hours.
- **Sectoral Impact**: Agriculture output and Youth Labor Force Participation exhibit distinct relationships with the underemployment metric that differ substantially from traditional total unemployment.

## Setup Requirements
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt # containing pandas, seaborn, scikit-learn, shap, statsmodels
```
## Advanced Diagnostic Visualizations
To supplement the primary dashboard, we developed deeper diagnostic visualizations (`Data_Analysis/Advanced_EDA_Dashboard.ipynb`) to robustly prove our econometric constraints before forecasting:
- **Demographic Disparity**: Highlights how female workers transition into underemployment at steeper angles than their male counterparts during shocks.
- **Time-Lagged Cross-Correlation (TLCC)**: Confirms the presence of extreme downstream delays (GDP/Inflation effects lagging by multiple quarters), strictly justifying the **VECM / ARDL** causal models over simpler multi-variate regressions.
- **Quarterly Outlier Stationarity**: Provides the base rationale for the structural tuning of hyperparameters in the **SARIMA** approach.

## Structural Break & Interaction Coefficients (RQ3)
Conforming to the latest checklist phase, the repository addresses RQ3 in `Data_Analysis/RQ3_Interaction_Report.ipynb`.
Taking the mathematical breakpoints found via our Zivot-Andrews testing (2021/2022 shock borders), we engineered crisis dummies and built OLS Interaction regressions (`Underemployment ~ GDP + Crisis_Dummy + GDP:Crisis_Dummy`). 

This quantifies the elasticity shifts, conclusively identifying how the statistical sensitivity of Sri Lanka's labor force towards inflation and GDP changed abruptly *after* the structural breaks.
