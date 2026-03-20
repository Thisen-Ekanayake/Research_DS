# Macro-Economic Drivers of Unemployment in Sri Lanka

This repository contains comprehensive data analysis on the macroeconomic and labor sector variables affecting the unemployment rate in Sri Lanka. It uses long-term (1991–2023) datasets from the World Bank and FRED to extrapolate insights leveraging both classical statistical correlations and advanced Machine Learning explanation algorithms (SHAP).

## Key Deliverables
1. **[Final Summary Visualizations](Data_Analysis/Final_Summary_Visualizations.ipynb)**: Start here. This main presentation dashboard contains the global correlation matrix, combined feature timelines, our SHAP feature-importance derivations, and time-lagged analytical findings.
2. **Individual Indicator Deep Dives (`/Data_Analysis`)**: Detailed univariate regression and probability distribution notebooks for exact sectors (Agriculture, Services, Youth LFPR, etc.) vs. Unemployment.
3. **Machine Learning Explanations**: Gradient Boosting and Random Forest algorithms natively implemented (`run_shap_analysis.py`, `lagged_analysis.py`) to dissect which parameters historically lead the highest predictive power for labor market shocks.

## Findings Summarized 
- **Inflation & Real GDP** possess the strongest generalized non-linear predictive boundaries when computing SHAP impacts mapping to the unemployment rate.
- **Services Sector & Youth LFPR** serve as incredibly high-signal direct continuous variables against the unemployment baseline.
- Time delays matter: Implementing a **2-year lag** on macro variables significantly boosts the variance explained by the Gradient Boosting model.

## Setup Requirements
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt # containing pandas, seaborn, scikit-learn, shap
```
