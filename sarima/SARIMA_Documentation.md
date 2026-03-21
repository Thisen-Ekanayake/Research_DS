# SARIMA Forecasting of Underemployment in Sri Lanka
## Technical Documentation

**Project:** Forecasting Underemployment in Sri Lanka Using Machine Learning  
**Institution:** Department of Computer Science & Engineering, University of Moratuwa  
**Module:** SARIMA Baseline Model with Auto-ARIMA (AICc Selection)  
**Authors:** Ekanayake T.N.D.S.W., Bulagala D.W.K.G., De Silva B.K.P., Pabasara W.G.K., Lelwala J.U.P.  
**Last Updated:** February 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [Theoretical Background](#3-theoretical-background)
   - 3.1 [What is ARIMA?](#31-what-is-arima)
   - 3.2 [Stationarity](#32-stationarity)
   - 3.3 [Model Order Selection with AICc](#33-model-order-selection-with-aicc)
   - 3.4 [Auto-ARIMA and the Hyndman–Khandakar Algorithm](#34-auto-arima-and-the-hyndmankhandakar-algorithm)
   - 3.5 [Walk-Forward Validation](#35-walk-forward-validation)
4. [Methodology Pipeline](#4-methodology-pipeline)
5. [Stationarity Testing](#5-stationarity-testing)
   - 5.1 [Augmented Dickey–Fuller (ADF) Test](#51-augmented-dickeyfuller-adf-test)
   - 5.2 [KPSS Test](#52-kpss-test)
   - 5.3 [Interpreting Conflicting Results](#53-interpreting-conflicting-results)
6. [ACF and PACF Analysis](#6-acf-and-pacf-analysis)
7. [Data Split Strategy](#7-data-split-strategy)
8. [Auto-ARIMA Model Selection](#8-auto-arima-model-selection)
9. [Residual Diagnostics](#9-residual-diagnostics)
10. [Walk-Forward Forecasting](#10-walk-forward-forecasting)
11. [Evaluation Metrics](#11-evaluation-metrics)
12. [Small Sample Considerations](#12-small-sample-considerations)
13. [Results Interpretation](#13-results-interpretation)
14. [Role in the Broader Project](#14-role-in-the-broader-project)
15. [File Structure](#15-file-structure)
16. [Dependencies](#16-dependencies)
17. [References](#17-references)

---

## 1. Project Overview

Sri Lanka's headline unemployment rate of approximately 3.8% conceals a significantly larger pool of labour market underutilisation. Underemployment — which encompasses workers constrained to fewer hours than desired (*time-related underemployment*) and workers employed below their qualification level (*skills-based underemployment*) — surged by approximately 80% between 2019 and 2022 in the wake of Sri Lanka's most severe post-independence economic crisis.

This notebook implements the **SARIMA baseline model**, the first of four forecasting models to be evaluated in the study. Its purpose is twofold:

1. To establish a **classical time-series benchmark** against which machine learning models (XGBoost, LSTM) are evaluated.
2. To characterise the temporal structure of the underemployment series — its trend, autocorrelation, and integration order — which informs the design of all subsequent models.

The SARIMA model is fitted using **automatic order selection via AICc** (the corrected Akaike Information Criterion), following the Hyndman–Khandakar stepwise algorithm implemented in the `pmdarima` library.

---

## 2. Dataset Description

The dataset is a **quarterly Labour Force Survey (LFS) aggregate** compiled from the following official sources:

| Source | Variables |
|--------|-----------|
| Department of Census & Statistics (DCS), Sri Lanka | Underemployment rate (%), gender disaggregation |
| Central Bank of Sri Lanka (CBSL) | Real GDP (LKR millions), GDP growth rate (%), inflation rate (%), exchange rate |
| ILO / ILOSTAT | Youth Labour Force Participation Rate (ages 15–24) |
| FAO / CEIC Data | Agricultural Production Index (composite and sub-indices) |
| World Bank DataBank | Remittance inflows (USD, % of GDP) |

**Coverage:** 2015–2024 (10 annual observations)  
**Dependent variable:** Composite underemployment rate (%)  
**Primary predictors used in this notebook:** GDP growth, inflation, youth LFPR, agricultural output index, remittances as % of GDP

> **Note on data frequency:** The full study targets quarterly LFS data (2015 Q1–2025 Q3, n≈43). The current dataset is annual (n=10), reflecting the publicly available aggregated LFS release cadence. All modelling decisions in this notebook account for this small-sample constraint.

### Key Variables

| Variable | Description | Unit |
|----------|-------------|------|
| `Underemployment_Rate` | Composite underemployment rate — primary dependent variable | % |
| `Underemployment_Male` | Male underemployment rate | % |
| `Underemployment_Female` | Female underemployment rate | % |
| `GDP_Growth` | Annual real GDP growth rate | % |
| `Inflation` | Annual inflation rate (CPI-based) | % |
| `Youth_LFPR` | Youth Labour Force Participation Rate (15–24) | % |
| `Agri_Output` | Composite agricultural production index | Index (base normalised) |
| `Remittances_pct_GDP` | Personal remittances received as % of GDP | % |

---

## 3. Theoretical Background

### 3.1 What is ARIMA?

ARIMA stands for **AutoRegressive Integrated Moving Average**. It is a family of models for univariate time-series forecasting that captures three types of structure:

- **AR (AutoRegressive, order *p*):** The current value of the series is a linear function of its own *p* lagged values. This captures momentum and mean-reversion dynamics.
- **I (Integrated, order *d*):** The series is differenced *d* times to achieve stationarity. A series with a unit root (random walk behaviour) requires *d*=1.
- **MA (Moving Average, order *q*):** The current value depends on *q* lagged forecast errors (residuals). This captures shock propagation.

A general ARIMA(*p*, *d*, *q*) model can be written as:

$$\phi(B)(1-B)^d y_t = c + \theta(B)\varepsilon_t$$

where:
- $B$ is the backshift operator: $B y_t = y_{t-1}$
- $\phi(B) = 1 - \phi_1 B - \dots - \phi_p B^p$ is the AR polynomial
- $\theta(B) = 1 + \theta_1 B + \dots + \theta_q B^q$ is the MA polynomial
- $(1-B)^d$ applies *d* rounds of differencing
- $\varepsilon_t \sim \mathcal{N}(0, \sigma^2)$ is white noise
- $c$ is an optional intercept/drift term

**SARIMA** (Seasonal ARIMA) extends this with additional seasonal AR and MA terms at lag *m* (the seasonal period). Since this study uses **annual data**, there is no sub-annual seasonality to model, so the seasonal components (*P*, *D*, *Q*)$_m$ are set to zero and the model reduces to a standard ARIMA.

### 3.2 Stationarity

A time series is **weakly stationary** if its mean, variance, and autocovariance structure do not change over time. This is a prerequisite for ARIMA modelling because the model assumes a stable data-generating process.

Most economic time series are non-stationary in levels — they exhibit trends, structural breaks, or unit roots. There are two common mechanisms:

- **Difference-stationary (DS):** The series has a stochastic trend (unit root). Taking first differences removes the trend. This corresponds to *d*=1 in ARIMA.
- **Trend-stationary (TS):** The series fluctuates around a deterministic trend. De-trending removes the non-stationarity.

For underemployment rates, which are bounded percentages, stationarity is plausible in levels, but this must be confirmed empirically through unit root tests rather than assumed.

### 3.3 Model Order Selection with AICc

Given a candidate set of ARIMA orders, the best model is selected by minimising the **corrected Akaike Information Criterion (AICc)**:

$$\text{AICc} = -2\ln(\hat{L}) + 2k + \frac{2k(k+1)}{n - k - 1}$$

where:
- $\hat{L}$ is the maximised likelihood of the fitted model
- $k$ is the number of estimated parameters (including intercept and $\sigma^2$)
- $n$ is the number of observations used in fitting

The standard AIC applies a penalty of $2k$ per parameter. AICc adds a **finite-sample correction** term $\frac{2k(k+1)}{n-k-1}$ that penalises complexity more heavily when $n$ is small. This correction matters significantly when $n/k < 40$, which is the case here (training set: $n=6$). **Using AIC without this correction on small samples systematically over-fits.**

The correction converges to AIC as $n \to \infty$, so AICc is recommended universally.

**Why not BIC?** The Bayesian Information Criterion (BIC) applies a larger penalty of $k \ln(n)$ and is consistent (selects the true model as $n \to \infty$ if it is in the candidate set). For small samples with uncertain true model order, AICc often achieves better out-of-sample predictive performance than BIC, which is why the proposal specifies AICc.

### 3.4 Auto-ARIMA and the Hyndman–Khandakar Algorithm

The `auto_arima` function implements the **Hyndman–Khandakar stepwise search** algorithm (Hyndman & Khandakar, 2008):

1. Determine the differencing order *d* using a unit root test (ADF or Phillips-Perron).
2. Starting from ARIMA(2,*d*,2) with drift, evaluate four variations: ARIMA(3,*d*,2), ARIMA(2,*d*,3), ARIMA(1,*d*,2), ARIMA(2,*d*,1).
3. At each step, move to the neighbour with the lowest AICc. If no neighbour improves AICc, the search terminates.
4. The algorithm also considers models with and without a constant term.

This stepwise approach is computationally efficient, evaluating far fewer models than an exhaustive grid search while typically finding the same optimal order. The `trace=True` parameter in the notebook prints the search trajectory for transparency.

### 3.5 Walk-Forward Validation

Standard train/test splits are **not valid for time series** because future observations cannot inform past predictions. The correct approach is **walk-forward validation** (also called time-series cross-validation or rolling-origin evaluation):

For each forecast horizon $t$ in the validation or test set:
1. Fit the model on all data up to time $t-1$.
2. Forecast one step ahead to $t$.
3. Record the error $e_t = y_t - \hat{y}_t$.
4. Move forward one period and repeat.

This preserves temporal order and provides an unbiased estimate of out-of-sample forecast accuracy. It is especially important here because the dataset includes the 2022 crisis period, and including post-crisis data in training would constitute lookahead bias with respect to pre-crisis validation.

---

## 4. Methodology Pipeline

```
Raw CSV Data
     │
     ▼
Data Loading & Cleaning
  • Strip whitespace from columns
  • Convert string-encoded numerics to float
  • Set annual DatetimeIndex (freq='YS')
     │
     ▼
Exploratory Data Analysis
  • Time series plot with crisis annotation
  • Gender disaggregation
  • GDP growth vs underemployment overlay
  • Correlation heatmap
     │
     ▼
Stationarity Tests
  • ADF (Augmented Dickey–Fuller)       → H0: unit root
  • KPSS (Kwiatkowski–Phillips–Schmidt–Shin) → H0: stationary
  • Tested on levels AND first difference
     │
     ▼
ACF / PACF Analysis
  • Identifies AR and MA signature patterns
  • Guides manual order bounds for auto-ARIMA
     │
     ▼
Train / Validation / Test Split
  • Train : 2015–2020  (n=6)
  • Val   : 2021–2022  (n=2)
  • Test  : 2023–2024  (n=2)
     │
     ▼
Auto-ARIMA (AICc)
  • Stepwise Hyndman–Khandakar search
  • Phillips-Perron test for d selection
  • AICc comparison table across 12 candidate orders
     │
     ▼
Residual Diagnostics
  • Residual time plot
  • Histogram + Q-Q plot (normality)
  • ACF of residuals (independence)
  • Ljung-Box portmanteau test
     │
     ▼
Walk-Forward Validation
  • One-step-ahead forecasts for 2021–2024
  • Retrain at each step
     │
     ▼
Evaluation Metrics
  • MAE, RMSE, MAPE, Directional Accuracy
     │
     ▼
2025 Projection
  • Refit on full dataset (2015–2024)
  • One-step-ahead forecast with 80% and 95% CI
```

---

## 5. Stationarity Testing

Before fitting any ARIMA model, we must determine the **integration order *d*** — the number of times the series must be differenced to achieve stationarity. This notebook applies two complementary tests.

### 5.1 Augmented Dickey–Fuller (ADF) Test

The ADF test evaluates the null hypothesis that the series has a **unit root** (i.e., it is non-stationary):

$$H_0: \delta = 0 \quad \text{(unit root present)}$$
$$H_1: \delta < 0 \quad \text{(stationary)}$$

The test augments the basic Dickey–Fuller regression with lagged difference terms to correct for serial correlation in the residuals. The number of lags is selected by AIC.

A **small p-value** (< 0.05) leads to rejection of $H_0$, suggesting the series is stationary. A **large p-value** indicates non-stationarity.

### 5.2 KPSS Test

The KPSS test (Kwiatkowski, Phillips, Schmidt & Shin, 1992) reverses the null hypothesis:

$$H_0: \text{series is stationary around a deterministic trend}$$
$$H_1: \text{series has a unit root}$$

A **small p-value** (< 0.05) leads to rejection of $H_0$, suggesting non-stationarity. A **large p-value** supports stationarity.

### 5.3 Interpreting Conflicting Results

| ADF result | KPSS result | Interpretation |
|------------|-------------|----------------|
| Reject $H_0$ (stationary) | Fail to reject $H_0$ (stationary) | Both agree: **stationary** → *d* = 0 |
| Fail to reject $H_0$ (unit root) | Reject $H_0$ (unit root) | Both agree: **non-stationary** → *d* = 1 |
| Reject $H_0$ | Reject $H_0$ | Conflicting: possibly **fractionally integrated** → conservative *d* = 1 |
| Fail to reject $H_0$ | Fail to reject $H_0$ | Conflicting: possibly **trend-stationary** → conservative *d* = 1 |

When tests conflict, the conservative choice of *d* = 1 is adopted to avoid spurious regression from any residual non-stationarity.

> **Small-sample caveat:** With only 10 observations, both ADF and KPSS tests have low statistical power. The results should be treated as indicative rather than conclusive. This is a known limitation of working with annual macroeconomic data.

---

## 6. ACF and PACF Analysis

Before running auto-ARIMA, the **Autocorrelation Function (ACF)** and **Partial Autocorrelation Function (PACF)** are plotted to visually characterise the series structure.

**ACF at lag *k*** measures the correlation between $y_t$ and $y_{t-k}$, including all indirect correlations through intermediate lags.

**PACF at lag *k*** measures the correlation between $y_t$ and $y_{t-k}$ after removing the effect of all shorter lags — the "direct" association.

The theoretical signatures used to manually identify ARIMA orders:

| Pattern | Suggested model |
|---------|-----------------|
| ACF cuts off after lag *q*; PACF tails off | MA(*q*) |
| PACF cuts off after lag *p*; ACF tails off | AR(*p*) |
| Both tail off gradually | ARMA(*p*, *q*) |
| ACF decays slowly / non-stationarity | Differencing needed |

These signatures serve as a sanity check on the auto-ARIMA output rather than as the primary model selection tool.

---

## 7. Data Split Strategy

The dataset is split into three non-overlapping, temporally ordered subsets:

| Subset | Period | Observations | Purpose |
|--------|--------|-------------|---------|
| **Train** | 2015–2020 | 6 | Model fitting and AICc selection |
| **Validation** | 2021–2022 | 2 | Hyperparameter tuning and model comparison |
| **Test** | 2023–2024 | 2 | Final unbiased performance evaluation |

This split is designed to:

1. Place the **2022 economic crisis** in the validation set, allowing models to be compared on their ability to handle the crisis period without having been trained on it.
2. Reserve 2023–2024 (post-IMF stabilisation) as a true holdout that no modelling decision has seen.
3. Mirror the structure proposed for the full quarterly dataset (Train: 2015 Q1–2022 Q2; Validation: 2022 Q3–2024 Q4; Test: 2025 Q1–Q3).

> The small size of each split (n=2) is a direct consequence of the annual data granularity. Quarterly data (n≈43) will yield more statistically meaningful split sizes of approximately 29/10/3 observations as specified in the proposal.

---

## 8. Auto-ARIMA Model Selection

The `auto_arima` function from `pmdarima` implements the Hyndman–Khandakar algorithm with the following configuration:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_p`, `max_q` | 2 | Constrained for small sample; higher orders are unidentifiable with n=6 |
| `max_d` | 1 | Annual data; second differencing destroys information |
| `information_criterion` | `'aicc'` | AICc preferred over AIC for small n; see Section 3.3 |
| `test` | `'pp'` | Phillips-Perron test; more robust than ADF for very small samples |
| `stepwise` | `True` | Hyndman–Khandakar stepwise search |
| `seasonal` | `False` | Annual frequency; no sub-annual seasonality present |

After auto-ARIMA selects the best model, a **supplementary AICc comparison table** is computed by explicitly fitting 12 standard candidate ARIMA orders using `statsmodels.SARIMAX`. This provides an auditable record of all models considered and allows manual verification of the auto-ARIMA selection.

**Why the Phillips-Perron test instead of ADF?**

The Phillips-Perron (PP) test uses a non-parametric correction for serial correlation rather than adding lagged differences. When the training set has only 6 observations, the ADF test requires estimating lag regression coefficients that leave too few degrees of freedom — it can fail with a `LinAlgError`. The PP test is numerically more stable in this regime while maintaining equivalent asymptotic properties.

---

## 9. Residual Diagnostics

A well-specified ARIMA model should produce residuals that behave like **white noise**: zero mean, constant variance, and no autocorrelation. Four diagnostics are applied:

### Residual Time Plot
Residuals plotted against time. Patterns (trends, heteroscedasticity, outliers) indicate model misspecification.

### Residual Histogram + Q-Q Plot
Assesses approximate normality of residuals. While ARIMA estimation does not strictly require Gaussian errors, normality is needed for prediction intervals to be valid. The Q-Q plot compares quantiles of the residuals against the standard normal — points should lie on the diagonal.

### ACF of Residuals
If significant autocorrelation remains in the residuals, the model has not fully captured the temporal structure, suggesting *p* or *q* should be increased.

### Ljung–Box Portmanteau Test
A formal test of the joint null hypothesis that the first *h* autocorrelations of the residuals are all zero:

$$H_0: \rho_1 = \rho_2 = \dots = \rho_h = 0 \quad \text{(residuals are white noise)}$$

$$Q_{LB} = n(n+2) \sum_{k=1}^{h} \frac{\hat{\rho}_k^2}{n-k} \sim \chi^2(h - p - q)$$

A **large p-value** (> 0.05) is desired — it indicates no significant residual autocorrelation and that the model adequately captures the serial dependence structure.

---

## 10. Walk-Forward Forecasting

The walk-forward procedure generates one-step-ahead forecasts for the validation and test periods. At each step:

1. All data up to year $t-1$ is used as the training window.
2. The ARIMA model (with the auto-ARIMA selected order) is refit from scratch on this window.
3. A one-step forecast for year $t$ is produced along with **80% prediction intervals**.
4. The actual value at $t$ is recorded, and the window expands by one observation.

**Prediction intervals** are derived from the forecast error variance of the SARIMAX state-space form. They widen as the forecast horizon increases and account for parameter estimation uncertainty in larger models (though not in the simplified `get_forecast` implementation, which conditions on estimated parameters).

The 80% interval is reported (rather than the conventional 95%) because with n=2 test observations, 95% intervals are so wide as to be uninformative. Policymakers also often prefer tighter intervals that reflect the most likely range.

---

## 11. Evaluation Metrics

Four metrics are reported for both validation and test sets:

### Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{n} \sum_{t=1}^{n} |y_t - \hat{y}_t|$$

Measures average absolute deviation in the original units (percentage points). Interpretable and robust to outliers. **Lower is better.**

### Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{t=1}^{n} (y_t - \hat{y}_t)^2}$$

Penalises large errors more heavily than MAE due to the squaring. Useful for detecting occasional large misses. **Lower is better.**

### Mean Absolute Percentage Error (MAPE)
$$\text{MAPE} = \frac{100}{n} \sum_{t=1}^{n} \left| \frac{y_t - \hat{y}_t}{y_t} \right|$$

Scale-free, expressed as a percentage. Facilitates comparison across models and datasets. **Lower is better.** Note: MAPE is undefined if any $y_t = 0$ and is asymmetric (it penalises over-forecasting more than under-forecasting).

### Directional Accuracy (DA)
$$\text{DA} = \frac{1}{n-1} \sum_{t=2}^{n} \mathbf{1}\left[\text{sign}(y_t - y_{t-1}) = \text{sign}(\hat{y}_t - \hat{y}_{t-1})\right] \times 100\%$$

Measures the proportion of periods for which the model correctly predicts the *direction* of change (increase or decrease). Particularly relevant for policy applications where the directionality of underemployment trends matters for early-warning systems. **Higher is better.** DA = 50% is equivalent to random guessing.

---

## 12. Small Sample Considerations

The annual dataset (n=10) imposes constraints that affect the entire modelling pipeline. These are documented transparently:

| Issue | Impact | Mitigation applied |
|-------|--------|--------------------|
| Low power of stationarity tests | ADF/KPSS results are indicative, not conclusive | Report both tests; apply conservative d=1 if conflicting |
| ADF test instability in auto-ARIMA | LinAlgError with training n=6 | Use Phillips-Perron test instead |
| High AICc penalty for complex models | AICc strongly prefers parsimonious orders | Expected: auto-ARIMA selects low *p*, *q* |
| Unstable parameter estimates for p,q > 2 | Inflated standard errors; near-singular Hessian | cap max_p=max_q=2 |
| Small validation/test sets (n=2 each) | High variance in metric estimates | Report metrics and interpret with caution |
| Wide prediction intervals | Forecast uncertainty is large relative to signal | Report 80% CI; note limitation |

> **Implication for the full study:** These limitations are inherent to the annual aggregation of the LFS data. The full quarterly dataset (n≈43) will substantially alleviate all of the above issues, enabling higher-order model search, more reliable unit root tests, and statistically meaningful cross-validation splits.

---

## 13. Results Interpretation

### Historical Series (2015–2024)

The underemployment rate fluctuates narrowly between 2.3% and 2.8% over the observed period. This compressed range reflects:

- The composite nature of the rate (blending time-related and qualification-based dimensions)
- The annual aggregation smoothing out quarterly spikes visible in the raw LFS micro-data
- The fact that the reported series may not fully capture the 2022 crisis peak of 8.3% cited in the proposal (which is a time-related underemployment quarterly peak, not the annual composite)

### Crisis Period (2020–2022)

The 2022 economic crisis — sovereign default, FX reserve collapse, inflation exceeding 70% — is annotated on all time series plots. The model validation set deliberately spans this period to test whether the ARIMA structure generalises under structural break conditions.

### 2025 Projection

The final model, refit on all 10 observations, generates a 2025 point forecast with 80% and 95% prediction intervals. The proposal's baseline projection of ~4.8% stabilisation by 2026 serves as an external benchmark for this forecast.

### Benchmarking Against ML Models

The SARIMA metrics (MAE, RMSE, MAPE, DA) form the **baseline** against which XGBoost and LSTM are evaluated. Per the proposal's protocol: if XGBoost outperforms LSTM by more than 2% MAPE on walk-forward validation, XGBoost is adopted as the primary ML model. The SARIMA MAPE will contextualise these ML comparisons.

---

## 14. Role in the Broader Project

This notebook is the **first of four model implementations** in the study:

```
Model 1: SARIMA          ← This notebook
Model 2: ARDL / VECM     (causal inference; econometric benchmark)
Model 3: XGBoost         (gradient boosting; primary ML model)
Model 4: LSTM            (recurrent neural network; sequential learning)
         │
         ▼
SHAP Interpretability (applied to XGBoost and LSTM)
         │
         ▼
Policy Brief (top-ranked predictors → targeted interventions)
```

The SARIMA model serves a specific role: it captures the **univariate temporal structure** of underemployment without using any economic predictors. By comparing SARIMA (no predictors) against ARDL/XGBoost/LSTM (with predictors), the study can quantify the marginal value of macroeconomic covariates for forecasting accuracy.

---

## 15. File Structure

```
project/
├── sarima_underemployment.ipynb    # Main SARIMA notebook (this codebase)
├── master_dataset.csv              # Compiled dataset (2015–2024, annual)
├── SARIMA_Documentation.md         # This documentation file
└── outputs/
    ├── eda_overview.png             # EDA: 4-panel exploratory plots
    ├── acf_pacf.png                 # ACF/PACF for levels and first difference
    ├── data_split.png               # Train/Val/Test split visualisation
    ├── residual_diagnostics.png     # 4-panel residual diagnostic plots
    ├── sarima_forecasts.png         # Walk-forward forecasts with CI
    └── forecast_2025.png            # 2025 projection with CI
```

---

## 16. Dependencies

| Library | Version (tested) | Purpose |
|---------|-----------------|---------|
| `pandas` | ≥ 1.5 | Data manipulation, DatetimeIndex |
| `numpy` | ≥ 1.23 | Numerical operations |
| `matplotlib` | ≥ 3.6 | Visualisation |
| `seaborn` | ≥ 0.12 | Statistical plot aesthetics |
| `statsmodels` | ≥ 0.14 | SARIMAX, ADF, KPSS, ACF/PACF, Ljung-Box |
| `pmdarima` | ≥ 2.0 | Auto-ARIMA with AICc |
| `scikit-learn` | ≥ 1.2 | MAE, RMSE metrics |
| `scipy` | ≥ 1.10 | Q-Q plot (probplot) |

Install all dependencies:

```bash
pip install pandas numpy matplotlib seaborn statsmodels pmdarima scikit-learn scipy
```

---

## 17. References

1. **Bell, D.N.F. & Blanchflower, D.G.** (2018). *Underemployment in the US and Europe*. NBER Working Paper No. 24927.

2. **Box, G.E.P., Jenkins, G.M., Reinsel, G.C. & Ljung, G.M.** (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.

3. **Central Bank of Sri Lanka** (2025). *Economic and Labour Market Indicators*. https://www.cbsl.gov.lk

4. **Department of Census and Statistics Sri Lanka** (2015–2025). *Labour Force Survey Quarterly Reports*. https://www.statistics.gov.lk

5. **Hyndman, R.J. & Khandakar, Y.** (2008). Automatic time series forecasting: the forecast package for R. *Journal of Statistical Software*, 27(3), 1–22.

6. **Hyndman, R.J. & Athanasopoulos, G.** (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. https://otexts.com/fpp3

7. **Kwiatkowski, D., Phillips, P.C.B., Schmidt, P. & Shin, Y.** (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. *Journal of Econometrics*, 54(1–3), 159–178.

8. **Pabasara, M.V.A.G.U. & Silva, H.P.T.N.** (2025). Impact of Selected Macroeconomic Variables on the Unemployment Rate in Sri Lanka (2005–2020). *International Journal of Social Statistics*, 2(1).

9. **Sugiura, N.** (1978). Further analysis of the data by Akaike's information criterion and the finite corrections. *Communications in Statistics — Theory and Methods*, 7(1), 13–26. *(Original AICc paper)*

10. **Akaike, H.** (1974). A new look at the statistical model identification. *IEEE Transactions on Automatic Control*, 19(6), 716–723.

---

*Documentation prepared for the research project "Forecasting Underemployment in Sri Lanka Using Machine Learning", University of Moratuwa, 2026.*
