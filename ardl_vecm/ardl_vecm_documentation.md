# ARDL/VECM Causal Inference — Technical Documentation

**Project:** Forecasting Underemployment in Sri Lanka Using Machine Learning  
**Phase:** 3 of 4 — Econometric Causal Inference  
**Authors:** Ekanayake T.N.D.S.W., Bulagala D.W.K.G., De Silva B.K.P., Pabasara W.G.K., Lelwala J.U.P.  
**Institution:** Department of Computer Science & Engineering, University of Moratuwa  
**Date:** March 2026

---

## Table of Contents

1. [Purpose and Scope](#1-purpose-and-scope)
2. [Data and Variables](#2-data-and-variables)
3. [Theoretical Framework](#3-theoretical-framework)
4. [Unit Root Testing — ADF and KPSS](#4-unit-root-testing--adf-and-kpss)
5. [Johansen Cointegration Test](#5-johansen-cointegration-test)
6. [Granger Causality Tests](#6-granger-causality-tests)
7. [ARDL Bounds Test](#7-ardl-bounds-test)
8. [VECM Estimation](#8-vecm-estimation)
9. [Model Diagnostics](#9-model-diagnostics)
10. [Results Summary and Interpretation Guide](#10-results-summary-and-interpretation-guide)
11. [Limitations](#11-limitations)
12. [Connection to Phase 4](#12-connection-to-phase-4)
13. [References](#13-references)

---

## 1. Purpose and Scope

This notebook constitutes Phase 3 of the research pipeline. Its objective is to establish and quantify the **causal relationships** between Sri Lanka's macroeconomic conditions and its underemployment rate, using classical time-series econometrics.

While Phase 4 (XGBoost + SHAP) answers *which predictors matter most for forecasting*, Phase 3 answers a structurally distinct question: *do these predictors causally precede movements in underemployment, and is there a stable long-run equilibrium between them?* This distinction is critical for policy — a variable can be predictively useful without being causally actionable, and vice versa.

The notebook implements the following sequence, which mirrors best practice in applied macroeconometrics:

```
Data preparation
      ↓
Unit root testing (ADF + KPSS)
      ↓
Cointegration decision (Johansen)
      ↓              ↓
  VECM path      ARDL bounds path
      ↓              ↓
Granger causality tests (both paths)
      ↓
Diagnostics
      ↓
Policy-relevant results summary
```

---

## 2. Data and Variables

### Dataset

- **Source:** `master_dataset.csv` — constructed in Phase 2 by merging DCS Labour Force Survey data with CBSL and World Bank macroeconomic series
- **Coverage:** Annual, 2015–2024 (n = 10 observations)
- **Imputation:** 2024 GDP growth rate is missing from the CBSL series at time of writing; it is linearly extrapolated from the 2021–2023 trend. This value should be replaced with the official CBSL estimate before final submission.

### Variables

| Role | Variable | Source | Units |
|------|----------|--------|-------|
| **Dependent** | `Underemployment_Rate` | DCS LFS | % of employed |
| Predictor | `GDP_Growth_Rate` | CBSL / World Bank | % YoY |
| Predictor | `Youth_LFPR_15_24` | DCS LFS | % of youth population |
| Predictor | `Informal_Pct` | DCS LFS | % of employment |
| Predictor | `Exchange_Rate_LKR_USD` | CBSL | LKR per USD |
| Predictor | `Inflation_Rate` | CBSL | % YoY CPI |
| Control | `Crisis_Dummy` | Derived | 1 if year = 2022, else 0 |

### Excluded Variables (Limitation)

Two predictors specified in the original research proposal are absent:

- **Remittance inflows (USD mn):** Consistent quarterly CBSL data for 2015–2024 was not sourced in time for Phase 2 dataset construction.
- **Agricultural output index:** FAO/CEIC quarterly series was not obtained. The `Informal_Pct` variable partially proxies the non-agricultural labour demand channel.

Both exclusions are documented formally in the notebook output and must be acknowledged in the paper's limitations section.

---

## 3. Theoretical Framework

### Why econometrics before machine learning?

Machine learning models are optimised for predictive accuracy — they learn patterns in data and generalise to unseen observations. Econometric models are optimised for causal inference — they test whether one variable *structurally precedes and drives* another, under explicit assumptions about data-generating processes. The two approaches answer different questions and reinforce each other: econometrics provides the causal narrative; ML provides the forecasting engine. This is why Phase 3 runs before Phase 4.

### The challenge of non-stationary macroeconomic series

Most macroeconomic time series — GDP, exchange rates, price levels — are non-stationary: their means and variances change over time. Regressing one non-stationary series on another without accounting for this produces **spurious regression**, where high R² and apparently significant coefficients are artefacts of shared trends rather than genuine economic relationships. The statistical solution is to determine the *order of integration* of each series before modelling.

A series is said to be **integrated of order d**, written I(d), if it must be differenced d times to become stationary. Most macroeconomic series are I(1) — stationary in first differences but not in levels. The modelling strategy depends entirely on what order of integration the data exhibit.

### The integration-cointegration-modelling decision tree

Once integration orders are established, two scenarios arise:

**Scenario A — No cointegration (or mixed I(0)/I(1)):** The ARDL bounds test is appropriate. It can handle regressors that are a mix of I(0) and I(1), does not require all variables to be the same order of integration, and tests for a long-run level relationship via an F-test on the lagged levels of all variables.

**Scenario B — All variables I(1) with cointegration:** The Johansen VECM framework is appropriate. Cointegrated variables share a common stochastic trend — they may drift apart in the short run but are bound together in the long run by an economic force. The VECM models both the long-run equilibrium and the short-run speed of adjustment back to it.

In this study, **both models are estimated**: ARDL as the primary model (given small n and mixed integration orders), and VECM as a robustness check (if Johansen indicates cointegration).

---

## 4. Unit Root Testing — ADF and KPSS

### Why two tests?

The Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests have opposite null hypotheses. Using only one test risks incorrect conclusions due to low power (ADF) or size distortions (KPSS). Using both in combination gives a more reliable determination of integration order.

| Test | Null Hypothesis (H₀) | Reject H₀ means... |
|------|---------------------|---------------------|
| ADF | Unit root present (non-stationary) | Series is stationary |
| KPSS | Series is stationary | Series has a unit root |

### Decision rule

| ADF result | KPSS result | Conclusion |
|------------|-------------|------------|
| Fail to reject H₀ (p > 0.05) | Reject H₀ (p < 0.05) | **I(1) — non-stationary** |
| Reject H₀ (p < 0.05) | Fail to reject H₀ (p > 0.05) | **I(0) — stationary** |
| Both reject or both fail to reject | — | **Ambiguous — examine first differences** |

### The Augmented Dickey-Fuller Test

The ADF test extends the basic Dickey-Fuller test by including lagged difference terms to account for serial correlation in residuals. The test regression is:

$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \varepsilon_t$$

where $p$ lagged differences are chosen by AIC. The null hypothesis is $\gamma = 0$ (unit root). The t-statistic on $\hat{\gamma}$ is compared against Dickey-Fuller critical values, which are more negative than standard t-distribution values because unit root distributions are skewed.

### The KPSS Test

The KPSS test decomposes the series into a deterministic trend, a random walk, and a stationary error:

$$y_t = \xi t + r_t + \varepsilon_t, \quad r_t = r_{t-1} + u_t$$

where $u_t \sim \text{i.i.d.}(0, \sigma_u^2)$. The null hypothesis is $\sigma_u^2 = 0$ (no random walk component — i.e., stationary). The LM test statistic is based on partial sums of residuals from a regression of $y_t$ on deterministic components.

### Small sample caveat

With n=10, both tests have low power and the lag selection is constrained to a maximum of 2. Results at levels should be treated as indicative. The first-difference results are typically more reliable; any variable that is stationary in first differences but not in levels is treated as I(1) for modelling purposes.

---

## 5. Johansen Cointegration Test

### The concept of cointegration

Two or more I(1) series are **cointegrated** if a linear combination of them is I(0). In economic terms: although each series wanders over time, they are bound together by a long-run equilibrium relationship. The classic example is the Fisher equation — even though interest rates and inflation each have unit roots, they tend to move together over long horizons.

For Sri Lanka's labour market, the hypothesis is that underemployment, GDP growth, youth LFPR, and the exchange rate share a stable structural relationship: when the economy contracts sharply (as in 2022), underemployment deviates from its equilibrium, but gradually reverts as conditions stabilise.

### The Johansen procedure

Johansen's (1988) maximum likelihood approach tests cointegration within a VAR framework. Consider a VAR(k) in levels:

$$Y_t = A_1 Y_{t-1} + A_2 Y_{t-2} + \cdots + A_k Y_{t-k} + \varepsilon_t$$

This can be rewritten in the VECM form:

$$\Delta Y_t = \Gamma_1 \Delta Y_{t-1} + \cdots + \Gamma_{k-1} \Delta Y_{t-k+1} + \Pi Y_{t-1} + \varepsilon_t$$

The matrix $\Pi = \alpha \beta'$ contains all the long-run information. Its **rank** determines the number of cointegrating relationships:

- $\text{rank}(\Pi) = 0$: no cointegration — model in first differences (VAR in differences)
- $\text{rank}(\Pi) = r$ where $0 < r < p$: $r$ cointegrating vectors — VECM is appropriate
- $\text{rank}(\Pi) = p$: all variables stationary — VAR in levels

$\beta$ is the matrix of **cointegrating vectors** (the long-run equilibrium coefficients) and $\alpha$ is the matrix of **adjustment speeds** (how fast each variable corrects back to equilibrium after a shock).

### Trace and maximum eigenvalue statistics

Johansen provides two test statistics. The **trace statistic** tests the null that the number of cointegrating vectors is at most $r$ against the alternative of $p$ vectors:

$$\lambda_{\text{trace}}(r) = -T \sum_{i=r+1}^{p} \ln(1 - \hat{\lambda}_i)$$

The **maximum eigenvalue statistic** tests $r$ versus $r+1$ cointegrating vectors specifically:

$$\lambda_{\max}(r, r+1) = -T \ln(1 - \hat{\lambda}_{r+1})$$

Both statistics are compared against asymptotic critical values. With n=10, these asymptotic values are approximate; the test is run with `det_order=0` (restricted constant, appropriate when variables have non-zero means but no deterministic trend) and `k_ar_diff=1`.

### Routing decision

The notebook sets a `USE_VECM` flag based on the trace statistic result at 5%. Regardless of this flag, both ARDL and VECM are estimated — the flag only determines which is labelled *primary* versus *robustness check*.

---

## 6. Granger Causality Tests

### What Granger causality means — and does not mean

**Granger (1969) causality** is a statement about *predictive precedence*, not structural causation. Variable X Granger-causes Y if past values of X contain information that improves the prediction of Y beyond what past values of Y alone provide. This is a necessary but not sufficient condition for true causality — a correlated third variable could drive both X and Y without either causing the other.

In the context of this study, finding that GDP growth Granger-causes underemployment means: knowing last year's GDP growth rate significantly improves our forecast of this year's underemployment rate. This is policy-relevant even without full structural identification, because it establishes the *timing* and *direction* of the relationship.

### Test specification

The standard Granger test compares two regressions:

**Restricted model** (Y predicted only by its own lags):
$$y_t = c + \sum_{i=1}^{p} \alpha_i y_{t-i} + \varepsilon_t$$

**Unrestricted model** (Y predicted by its own lags and X lags):
$$y_t = c + \sum_{i=1}^{p} \alpha_i y_{t-i} + \sum_{i=1}^{p} \beta_i x_{t-i} + u_t$$

The null hypothesis H₀: $\beta_1 = \beta_2 = \cdots = \beta_p = 0$ (X does not Granger-cause Y) is tested via an F-statistic:

$$F = \frac{(RSS_R - RSS_{UR}) / p}{RSS_{UR} / (n - 2p - 1)}$$

### Multiple testing correction — Bonferroni

Five predictors are tested simultaneously. Without correction, running five tests at α=0.05 each gives a family-wise error rate (probability of at least one false positive) of approximately 1 − 0.95⁵ ≈ 23%. The **Bonferroni correction** divides the significance threshold by the number of tests:

$$\alpha_{\text{Bonferroni}} = \frac{0.05}{5} = 0.01$$

Results are reported at both the nominal level (α=0.05) and the Bonferroni-corrected level (α=0.01). Variables significant at Bonferroni are labelled as strong Granger causes; those significant at nominal only are noted as suggestive.

### Lag selection

With n=10, maximum lag=2 is used. Lag=1 is the primary result; lag=2 provides confirmatory evidence if the degrees of freedom permit. This is a conservative choice — increasing lags would consume too many degrees of freedom and produce unreliable F-statistics.

---

## 7. ARDL Bounds Test

### Motivation — handling mixed integration orders

A key practical advantage of the **Autoregressive Distributed Lag (ARDL)** framework (Pesaran, Shin & Smith, 2001) is that it does not require all variables to be the same order of integration. It can accommodate a mix of I(0) and I(1) regressors, which is common in small macroeconomic datasets where unit root tests give ambiguous results.

The ARDL bounds test also performs better in small samples than Johansen's VECM, making it the primary model in this study.

### ARDL model specification

The unrestricted ARDL(p, q₁, q₂, ..., qₖ) model for k regressors is:

$$\Delta y_t = c + \sum_{i=1}^{p} \phi_i \Delta y_{t-i} + \sum_{j=1}^{k} \sum_{i=0}^{q_j} \theta_{ji} \Delta x_{jt-i} + \lambda_0 y_{t-1} + \sum_{j=1}^{k} \lambda_j x_{j,t-1} + \varepsilon_t$$

The lagged *levels* terms $\lambda_0 y_{t-1} + \sum \lambda_j x_{j,t-1}$ form the **error correction mechanism** (ECM). Their joint significance tests whether a long-run level relationship exists.

### The bounds test F-statistic

The bounds test examines H₀: $\lambda_0 = \lambda_1 = \cdots = \lambda_k = 0$ (no long-run relationship) using a standard F-statistic. Pesaran et al. (2001) provide two sets of asymptotic critical values:

- **I(0) lower bound:** assumes all regressors are stationary
- **I(1) upper bound:** assumes all regressors have unit roots

The decision rules are:

| F-statistic | Conclusion |
|-------------|------------|
| F > I(1) upper bound | Long-run relationship confirmed |
| F < I(0) lower bound | No long-run relationship |
| I(0) bound < F < I(1) bound | Inconclusive — integration order matters |

For k=5 predictors with unrestricted intercept and no trend, the Pesaran (2001) critical values are:

| Significance | I(0) lower | I(1) upper |
|-------------|-----------|-----------|
| 10% | 2.26 | 3.35 |
| 5% | 2.62 | 3.79 |
| 1% | 3.41 | 4.68 |

### Lag order selection

The notebook uses `ardl_select_order()` with AIC minimisation, constrained to a maximum of 1 lag per variable. AIC balances model fit against parameter parsimony — important given the small sample. A fallback to robust OLS (HC3 heteroskedasticity-consistent standard errors) is implemented if the ARDL module cannot achieve convergence given the degrees-of-freedom constraint.

### Long-run coefficients

If the bounds test confirms a long-run relationship, the long-run coefficients are recovered as:

$$\hat{\theta}_j = -\frac{\hat{\lambda}_j}{\hat{\lambda}_0}$$

These coefficients represent the equilibrium elasticity: the expected change in the underemployment rate for a one-unit permanent change in predictor $j$, once the system has fully adjusted.

---

## 8. VECM Estimation

### The Vector Error Correction Model

The VECM is appropriate when multiple I(1) variables share cointegrating relationships. It models the *short-run dynamics* (how variables respond to shocks quarter-to-quarter) and the *long-run adjustment* (how fast the system corrects deviations from equilibrium).

The VECM of order k for a system of p variables with r cointegrating vectors is:

$$\Delta Y_t = \alpha \beta' Y_{t-1} + \sum_{i=1}^{k-1} \Gamma_i \Delta Y_{t-i} + \varepsilon_t$$

where:

- $\beta'Y_{t-1}$ is the **error correction term** (ECT) — the deviation from long-run equilibrium at time t−1
- $\alpha$ is the **adjustment matrix** — the speed and direction at which each variable corrects toward equilibrium
- $\Gamma_i$ captures short-run dynamic interactions between variables

### The cointegrating vector β

The cointegrating vector $\beta$ defines the long-run equilibrium relationship. Normalised on underemployment:

$$\text{Underemployment}_t = \beta_1 \cdot \text{GDP Growth}_t + \beta_2 \cdot \text{Youth LFPR}_t + \beta_3 \cdot \text{Informal}_t + \beta_4 \cdot \text{Exchange Rate}_t + \beta_5 \cdot \text{Inflation}_t + \mu$$

This relationship is not estimated by OLS but by maximum likelihood, which accounts for the joint determination of all variables.

### The adjustment coefficient α

The $\alpha$ coefficient for the underemployment equation is the most policy-relevant output from the VECM. A **negative and significant** $\alpha$ for underemployment means the variable is error-correcting: when underemployment is above its long-run equilibrium (positive ECT), the negative $\alpha$ pulls it back down in the next period.

The magnitude of $\alpha$ indicates the *speed of adjustment*: an $\alpha$ of −0.5 means 50% of any disequilibrium is corrected within one year. An $\alpha$ close to 0 suggests the variable adjusts very slowly; an $\alpha$ greater than 1 in absolute value suggests overshooting.

### Why VECM is secondary in this study

The VECM has significant demands on sample size. With 6 variables (1 dependent + 5 predictors) and rank r=1, a VECM(1) consumes approximately $p(p \cdot k + r) = 6(6 + 1) = 42$ parameters — comparable to or exceeding the number of observations after differencing. The model is estimated for completeness and theoretical grounding, but its coefficient estimates are unreliable at n=10. The ARDL bounds test is the primary causal output for this study.

---

## 9. Model Diagnostics

Four diagnostic tests are run on the ARDL residuals to validate that the model's statistical inferences are reliable.

### Durbin-Watson test (serial correlation)

The Durbin-Watson statistic tests for first-order autocorrelation in residuals. A value near 2 indicates no autocorrelation; values below 1.5 suggest positive autocorrelation; values above 2.5 suggest negative autocorrelation. Serial correlation does not bias coefficient estimates but invalidates standard errors and test statistics, making hypothesis tests unreliable.

$$DW = \frac{\sum_{t=2}^{n}(\hat{\varepsilon}_t - \hat{\varepsilon}_{t-1})^2}{\sum_{t=1}^{n}\hat{\varepsilon}_t^2}$$

### Ljung-Box Q-test (higher-order autocorrelation)

The Ljung-Box test generalises the Durbin-Watson to test multiple lags simultaneously. The null hypothesis is that all autocorrelations up to lag h are zero:

$$Q = n(n+2) \sum_{k=1}^{h} \frac{\hat{\rho}_k^2}{n-k} \sim \chi^2(h)$$

Significant Q-statistics at lags 1 or 2 indicate residual structure that the model has not captured — potentially requiring additional lags or a different specification.

### Breusch-Pagan test (heteroskedasticity)

The Breusch-Pagan test checks whether residual variance is constant (homoskedasticity). The null is homoskedasticity. Rejection means the OLS standard errors are inefficient, and HC3 robust standard errors should be used instead. In macroeconomic data, heteroskedasticity often occurs around structural breaks such as the 2022 crisis — the variance of underemployment shocks during the crisis period may differ from normal years.

### Jarque-Bera test (normality)

The Jarque-Bera test examines whether residuals are normally distributed using skewness and excess kurtosis:

$$JB = \frac{n}{6}\left(S^2 + \frac{(K-3)^2}{4}\right) \sim \chi^2(2)$$

where S is sample skewness and K is sample kurtosis. Non-normality does not invalidate OLS estimates (which rely on asymptotic theory) but does affect the validity of small-sample t and F tests. Given n=10, non-normal residuals are common and are noted as a limitation rather than a model failure.

---

## 10. Results Summary and Interpretation Guide

### Reading the Granger causality output

The output table lists each predictor with its F-statistic, p-value at lag=1, and whether it passes the Bonferroni-corrected threshold (α=0.01). In line with the research proposal's expected ranking:

> Youth LFPR > Services Sector Share > GDP Growth > Agricultural Output > Remittances

Variables passing Bonferroni correction are strong Granger causes and should be highlighted in the paper as primary economic drivers. Variables passing only nominal α=0.05 are reported as suggestive, supporting discussion but not the central causal claim.

### Reading the ARDL bounds test output

The F-statistic from the bounds test is compared against Pesaran et al. (2001) critical values for k=5. The three possible outcomes map to the following paper statements:

- **F > I(1) upper bound:** *"The bounds test confirms a long-run cointegrating relationship between underemployment and the macroeconomic predictors at the 5% significance level (F = X.XX > 3.79)."*
- **F < I(0) lower bound:** *"The bounds test finds no evidence of a long-run level relationship. The predictors influence underemployment through short-run dynamics only."*
- **Inconclusive:** *"The bounds test F-statistic falls within the inconclusive band. Given the mixed integration orders observed, ARDL short-run coefficients remain informative but long-run inference is treated with caution."*

### Reading the VECM alpha (adjustment) coefficients

A negative and significant $\hat{\alpha}$ for the underemployment equation confirms error-correcting behaviour. The policy interpretation: *"Following the 2022 economic shock, underemployment deviated from its long-run equilibrium. The estimated adjustment coefficient of α = X.XX suggests that approximately [|α| × 100]% of this disequilibrium was corrected within one year."*

### The 2022 measurement anomaly

A recurring finding in this dataset is that the measured underemployment rate *decreased* to 2.3% in 2022 despite the most severe economic crisis in Sri Lanka's post-independence history. This counterintuitive result is a known measurement artefact: during acute economic crisis with extreme inflation and cash shortage, workers accept any available hours rather than reporting themselves as wanting more work. The LFS instrument captures *stated* desire for more hours, which is suppressed when workers fear that admitting underemployment could cost them their current position. This is acknowledged in all model outputs as a data limitation and discussed in the paper's methodology section.

---

## 11. Limitations

### Sample size
The most significant constraint in this study is n=10 annual observations. Johansen's cointegration test and the VECM rely on asymptotic theory — their critical values assume large samples. At n=10, both tests are likely to be over-fitted, and coefficient estimates from the VECM are unreliable. All econometric results from Phase 3 are therefore treated as directional and cross-validated against the XGBoost/SHAP findings in Phase 4. The proposal's quarterly LFS data (n≈43 by 2025 Q3) would substantially improve statistical reliability, but the current master dataset is constructed at annual frequency due to data availability constraints.

### Missing variables
The two variables excluded (remittances and agricultural output index) cover the channels most connected to Sri Lanka's economic vulnerability: external financial flows and agricultural sector cyclicality. Their absence likely attenuates the model's ability to explain crisis-period underemployment dynamics. Future work should incorporate both once quarterly CBSL/FAO data are obtained.

### 2024 GDP imputation
The GDP growth rate for 2024 is linearly extrapolated from the 2021–2023 trend. This introduces measurement error into the most recent observation, which is also in the Phase 4 test set. The actual CBSL 2024 Annual Report figure should replace this value before Phase 4 modelling.

### Annual vs quarterly mismatch
The research proposal specifies quarterly LFS data (2015 Q1 to 2025 Q3, n≈43). Quarterly data would enable more reliable unit root testing, richer lag structures in ARDL/VECM, and improved SHAP temporal dynamics in Phase 4. The annual dataset represents a pragmatic compromise based on what could be consistently merged across data sources.

---

## 12. Connection to Phase 4

Phase 3 produces three outputs that directly inform Phase 4 model design:

**1. Variable selection signal from Granger tests**  
Predictors that fail Granger causality at both nominal and Bonferroni levels are candidates for exclusion from the XGBoost feature set or downweighting via L1 regularisation. Predictors with strong Granger evidence should be prioritised in SHAP analysis.

**2. Long-run coefficient signs from ARDL/VECM**  
The direction of ARDL long-run coefficients provides a benchmark for SHAP's directionality. If ARDL finds a negative long-run relationship between GDP growth and underemployment, XGBoost SHAP values for GDP growth should show predominantly negative contributions. Systematic disagreement between Phase 3 and Phase 4 directional findings would warrant investigation.

**3. Crisis period framing**  
The 2022 crisis dummy is retained in Phase 4 as an engineered feature (alongside GDP growth × crisis interaction terms). The ARDL/VECM evidence on how quickly the economy corrected post-2022 informs the selection of lag features in the XGBoost input matrix.

---

## 13. References

Bell, D.N.F. and Blanchflower, D.G. (2018). *Underemployment in the US and Europe.* NBER Working Paper No. 24927.

Department of Census and Statistics Sri Lanka (2015–2025). *Labour Force Survey Quarterly Reports.* https://www.statistics.gov.lk

Granger, C.W.J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3), 424–438.

Johansen, S. (1988). Statistical analysis of cointegrating vectors. *Journal of Economic Dynamics and Control*, 12(2–3), 231–254.

Kwiatkowski, D., Phillips, P.C.B., Schmidt, P. and Shin, Y. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. *Journal of Econometrics*, 54(1–3), 159–178.

MacKinnon, J.G. (1996). Numerical distribution functions for unit root and cointegration tests. *Journal of Applied Econometrics*, 11(6), 601–618.

Pabasara, M.V.A.G.U. and Silva, H.P.T.N. (2025). Impact of Selected Macroeconomic Variables on the Unemployment Rate in Sri Lanka (2005–2020). *International Journal of Social Statistics*, 2(1).

Pesaran, M.H., Shin, Y. and Smith, R.J. (2001). Bounds testing approaches to the analysis of level relationships. *Journal of Applied Econometrics*, 16(3), 289–326.

Said, S.E. and Dickey, D.A. (1984). Testing for unit roots in autoregressive-moving average models of unknown order. *Biometrika*, 71(3), 599–607.

---

*This document is part of the Phase 3 output for the University of Moratuwa research project on forecasting underemployment in Sri Lanka. It should be read alongside `ardl_vecm_causal_inference.ipynb` (the executable notebook) and `master_dataset.csv` (the underlying data).*
