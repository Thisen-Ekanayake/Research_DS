# Documentation
## Analysing Economic Drivers of Underemployment in Sri Lanka (2015–2024)

---

## 1. Combined Dataset

**File:** `sri_lanka_labour_macro_combined.csv`  
**Coverage:** 2015–2024 (10 annual observations)  
**Columns:** 20

### Column Reference

| Column | Unit | Source |
|---|---|---|
| `year` | — | — |
| `underemployment_male_pct` | % of male employment | DCS LFS |
| `underemployment_female_pct` | % of female employment | DCS LFS |
| `underemployment_total_pct` | % of total employment | DCS LFS |
| `unemp_sector_agri` | % underemployed in agriculture | DCS LFS |
| `unemp_sector_industry` | % underemployed in industry | DCS LFS |
| `unemp_sector_services` | % underemployed in services | DCS LFS |
| `unemp_sector_total` | % overall (sector breakdown) | DCS LFS |
| `tru_female` | % of female employment | ILO / World Bank |
| `tru_male` | % of male employment | ILO / World Bank |
| `gdp_usd` | Current USD | FRED / World Bank |
| `gdp_growth_pct` | % year-on-year | Derived |
| `real_gdp_mn_usd` | Millions USD (constant prices) | FRED |
| `inflation_pct` | % consumer prices | FRED / World Bank |
| `agri_output_index` | Index (base 2014–2016 = 100) | FAO |
| `services_employment_share_pct` | % of total employment | ILO (modelled) |
| `youth_lfpr_female_pct` | % of female population 15–24 | ILO (modelled) |
| `remittance_usd` | Current USD | World Bank |
| `crisis_dummy` | 0 / 1 | Derived (1 = 2022–2023) |
| `period` | pre-crisis / crisis / recovery | Derived |

### Data Notes

- **TRU (time-related underemployment)** data from ILO/World Bank only covers 2020–2023; values are `NaN` outside this range.
- **Real GDP** is not available for 2024 in the source; that cell is `NaN`.
- `crisis_dummy` flags 2022 and 2023 as the crisis period, consistent with Sri Lanka's sovereign default and IMF stabilisation timeline.
- The `period` column classifies years as: **pre-crisis** (2015–2019), **crisis** (2020–2023), **recovery** (2024–).
- Agricultural output index is averaged across all crop/livestock categories under the FAO "Agriculture" aggregate item.

---

## 2. Structural Break Tests

### 2.1 Why Test for Structural Breaks?

Standard time-series models assume stable relationships between variables over time. Economic shocks — such as a sovereign default, a pandemic, or a currency crisis — can permanently alter those relationships. Structural break tests detect the point in time where such a shift occurred, and whether it was statistically significant. Without identifying breaks, regression results can be spurious or misleading.

---

### 2.2 Zivot-Andrews Test

The **Zivot-Andrews (ZA) test** is an extension of the Augmented Dickey-Fuller (ADF) unit root test. The standard ADF test assumes no structural breaks exist, which causes it to under-reject the null hypothesis of a unit root when a break is present — making a stationary series appear non-stationary.

ZA fixes this by endogenously searching for the breakpoint. For each candidate break year, it fits a model allowing a one-time shift in the intercept, the trend, or both (Model C). The break year that produces the most negative t-statistic — the point where the series looks most stationary conditional on a break — is selected as the estimated break date.

**Null hypothesis:** The series has a unit root with no structural break.  
**Rejection** of the null means the series is stationary around a broken trend — i.e., a real structural shift occurred.

**Critical values (Model C):**

| Significance | Critical Value |
|---|---|
| 1% | −5.57 |
| 5% | −5.08 |
| 10% | −4.82 |

**Results Summary:**

| Series | Detected Break | t-statistic | Significant |
|---|---|---|---|
| Underemployment Rate | 2021 | −5.47 | ✅ ** |
| GDP Growth | 2022 | −4.91 | ✅ * |
| Inflation | 2022 | −17.62 | ✅ *** |
| Agricultural Output Index | 2020 | −3.45 | ❌ |
| Services Employment Share | 2018 | −3.97 | ❌ |
| Youth LFPR | 2018 | −3.29 | ❌ |

Underemployment, GDP growth, and inflation all show statistically significant breaks centred on 2021–2022, confirming that the economic crisis structurally altered these series. Agricultural output, services share, and youth LFPR show no significant single break — their movements are better characterised as gradual trends.

---

### 2.3 Bai-Perron Test

The **Bai-Perron (BP) test** extends break detection to allow multiple simultaneous breakpoints. Rather than assuming one break and searching for it (as ZA does), BP estimates the optimal number and location of breaks jointly, minimising the total sum of squared residuals across all segments.

Two algorithms are used here:

- **PELT (Pruned Exact Linear Time):** Efficiently finds the globally optimal set of breakpoints using a dynamic programming approach with a penalty term that controls against overfitting. Suitable for detecting mean shifts.
- **BinSeg (Binary Segmentation):** A faster approximation that recursively splits the series at the most significant break. Less precise than PELT but useful as a cross-check.

**Results Summary:**

| Series | PELT Breaks | Notes |
|---|---|---|
| Underemployment Rate | 2022 | Aligns with crisis peak |
| GDP Growth | 2021 | Pre-crisis contraction start |
| Inflation | 2021 | Inflation surge preceded default |
| Agricultural Output Index | none | No structural shift detected |
| Services Employment Share | 2021 | Gradual, confirmed by PELT |
| Youth LFPR | 2019 | Pre-pandemic shift |

---

### 2.4 Interpreting ZA and BP Together

The two tests are complementary. ZA tells you whether a break was large enough to render a series stationary — it is primarily a unit root test. BP tells you where the mean of a series shifted, regardless of stationarity implications.

When both tests agree on a break year (e.g., GDP Growth: ZA=2022, BP=2021), the evidence for a structural shift in that year range is strong. When only one detects a break (e.g., Services Share: ZA=2018 insignificant, BP=2021), the shift is real but smaller in magnitude.

**Key conclusion:** The 2021–2022 window is confirmed as a statistically significant structural break in Sri Lanka's underemployment rate, GDP growth, and inflation. This validates the use of a crisis dummy in subsequent regression and ARDL/VECM modelling.
