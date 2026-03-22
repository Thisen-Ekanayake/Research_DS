# 2025 Data Estimation — Sri Lanka Labour & Macro Dataset

**Document:** Methodology, Data Sources, and Availability  
**Dataset:** `sri_lanka_labour_macro_combined_2025.csv`  
**Script:** `estimate_2025.py`  
**Prepared:** March 2026  
**Coverage:** Estimation of year 2025 values for all 19 columns in the dataset

---

## Overview

The original dataset covers 2015–2024. A 2025 row was constructed using three tiers of evidence:

| Tier | Label | Description |
|------|-------|-------------|
| 1 | **Official** | Value confirmed in an official published source (DCS, CBSL) |
| 2 | **Partial** | Derived from 2–3 available quarterly values; Q4 not yet published |
| 3 | **Estimated** | No direct 2025 data available; derived via trend extrapolation or ratio method |

All estimation logic is reproducible via `estimate_2025.py`.

---

## Estimation Methods Used

### Method A — Official value
Value taken directly from a confirmed official government or central bank publication. No modelling applied.

### Method B — Quarterly average
Where DCS LFS quarterly bulletins exist for Q1–Q3 2025, a simple arithmetic mean is computed across available quarters. Q4 (not yet released as of March 2026) is estimated using historical seasonal patterns from the same quarter in prior years.

### Method C — Ratio scaling
A Q1 2025 ratio between a sub-indicator (e.g., agricultural unemployment) and the total is computed from the LFS Q1 bulletin. This ratio is then applied to the annual total unemployment average to estimate the sub-sector annual average.

### Method D — OLS trend extrapolation
Where no 2025 data is available, an ordinary least squares (OLS) linear trend is fitted on the prior 3 years (2022–2024) and extrapolated one period forward. A floor value is applied to prevent implausible results below historical minima.

### Method E — Growth-rate application
A known growth rate (e.g., GDP, agriculture) is applied multiplicatively to the prior year's index or value.

### Method F — Logical classification
Non-numeric categorical variables assigned based on economic context (e.g., `period = "recovery"`, `crisis_dummy = 0`).

---

## Column-by-Column Documentation

---

### `gdp_growth_pct`

| | |
|---|---|
| **2025 value** | `5.0` |
| **Availability** | ✅ Available |
| **Method** | A — Official value |
| **Source** | Department of Census and Statistics (DCS), National Accounts Q4 2025 press release |
| **Notes** | DCS confirmed annual 2025 real GDP growth at 5.0%. Q4 2025 growth was 4.8%; annual average confirmed at 5.0%. |

---

### `gdp_usd`

| | |
|---|---|
| **2025 value** | `~105,989,571,680` (≈ $106 bn) |
| **Availability** | 🟡 Partial |
| **Method** | E — Growth-rate application |
| **Source** | IMF WEO October 2025; DCS National Accounts; CBSL exchange rate data |
| **Notes** | No official full-year 2025 nominal USD GDP published as of March 2026. Estimated by applying 5.0% real growth + 2.0% GDP deflator (consistent with low inflation environment) to 2024 nominal GDP ($98.96 bn). IMF WEO projects ~$105–108 bn. |

---

### `real_gdp_mn_usd`

| | |
|---|---|
| **2025 value** | `44,503.65` |
| **Availability** | 🟡 Partial |
| **Method** | E — Growth-rate application + exchange rate conversion |
| **Source** | DCS National Accounts (LKR constant 2015 prices); CBSL avg LKR/USD for 2025 |
| **Notes** | DCS published real GDP at constant 2015 LKR prices: LKR 13,128,577 mn. Converted at avg 2025 exchange rate of ~295 LKR/USD. Note: this column appears to track a LKR-denominated constant-price series expressed in USD-equivalent; values are consistent with prior years in the dataset when using this conversion rate. |

---

### `inflation_pct`

| | |
|---|---|
| **2025 value** | `0.433` |
| **Availability** | 🟡 Partial |
| **Method** | B — Arithmetic average of monthly official CCPI YoY readings |
| **Source** | Central Bank of Sri Lanka (CBSL), monthly CCPI press releases, January–December 2025 |
| **Notes** | Monthly YoY CCPI (2021=100) readings used: Jan −4.0%, Feb −2.0%, Mar −1.0%, Apr +0.5%, May +1.0%, Jun +1.2%, Jul +0.9%, Aug +1.1%, Sep +1.5%, Oct +1.8%, Nov +2.1%, Dec +2.1%. Annual average = 0.43%. The wide swing reflects strong base effects from the 2022 crisis peak. |

---

### `unemp_sector_total`

| | |
|---|---|
| **2025 value** | `3.975` |
| **Availability** | 🟡 Partial |
| **Method** | B — Quarterly average with Q4 seasonal estimate |
| **Source** | DCS LFS Quarterly Bulletins: Q1 2025, Q2 2025, Q3 2025 |
| **Notes** | Q1 = 3.8%, Q2 = 3.8%, Q3 = 4.3% (official). Q4 estimated at 4.0% based on historical Q3→Q4 decline (e.g., 2023 Q3 = 3.7%, Q4 = 4.0%). Annual average: 3.975%. Q4 2025 bulletin not yet released as of March 2026. |

---

### `unemp_sector_agri`

| | |
|---|---|
| **2025 value** | `5.753` |
| **Availability** | 🟡 Partial |
| **Method** | C — Ratio scaling from Q1 2025 LFS bulletin |
| **Source** | DCS LFS Q1 2025 Quarterly Bulletin (Table 8, sectoral breakdown) |
| **Notes** | Q1 2025 LFS: agricultural unemployment = 5.5%, total = 3.8%. Ratio = 1.447. Applied to annual total average (3.975%) to yield 5.75%. Annual sectoral breakdown not published until annual LFS report. |

---

### `unemp_sector_industry`

| | |
|---|---|
| **2025 value** | `3.661` |
| **Availability** | 🟡 Partial |
| **Method** | C — Ratio scaling from Q1 2025 LFS bulletin |
| **Source** | DCS LFS Q1 2025 Quarterly Bulletin |
| **Notes** | Q1 2025: industry unemployment = 3.5%, total = 3.8%. Ratio = 0.921. Applied to annual total average. |

---

### `unemp_sector_services`

| | |
|---|---|
| **2025 value** | `2.615` |
| **Availability** | 🟡 Partial |
| **Method** | C — Ratio scaling from Q1 2025 LFS bulletin |
| **Source** | DCS LFS Q1 2025 Quarterly Bulletin |
| **Notes** | Q1 2025: services unemployment = 2.5%, total = 3.8%. Ratio = 0.658. Applied to annual total average. |

---

### `tru_female`

| | |
|---|---|
| **2025 value** | `5.775` |
| **Availability** | 🟡 Partial |
| **Method** | B — Quarterly average with interpolation |
| **Source** | DCS LFS Q1 2025 Bulletin (Table 8); Adaderana Biz, Q3 2025 LFS coverage (January 21, 2026) |
| **Notes** | Q1 female unemployment = 6.3% (official). Q3 2025 shows improving female participation (LFPR rose to 33.9%). Q2 estimated at 5.8%, Q3 at 5.5%, Q4 at 5.5% (declining trend). Annual average: 5.775%. |

---

### `tru_male`

| | |
|---|---|
| **2025 value** | `2.75` |
| **Availability** | 🟡 Partial |
| **Method** | B — Quarterly average with interpolation |
| **Source** | DCS LFS Q1 2025 Bulletin (Table 8); Adaderana Biz, Q3 2025 LFS coverage |
| **Notes** | Q1 male unemployment = 2.5% (official). Q3 total rate rose to 4.3% with male LFPR at 68.6%. Q2 estimated at 2.7%, Q3 at 3.0%, Q4 at 2.8%. Annual average: 2.75%. |

---

### `underemployment_total_pct`

| | |
|---|---|
| **2025 value** | `2.367` |
| **Availability** | ❌ Not available (estimated) |
| **Method** | D — OLS trend extrapolation (2022–2024) |
| **Source** | Prior dataset values (2022 = 2.7%, 2023 = 3.7%, 2024 = 2.2%); DCS LFS Annual Reports |
| **Notes** | Annual LFS report for 2025 not yet published. OLS trend on 3 years yields 2.37%. Floor of 1.5% applied. Note: 2023 spike (3.7%) reflects peak crisis underutilisation; 2025 recovery expected to normalise. |

---

### `underemployment_male_pct`

| | |
|---|---|
| **2025 value** | `2.267` |
| **Availability** | ❌ Not available (estimated) |
| **Method** | D — OLS trend extrapolation (2022–2024) |
| **Source** | Prior dataset values; DCS LFS Annual Reports |
| **Notes** | Annual LFS report for 2025 not yet published. Floor of 1.2% applied. |

---

### `underemployment_female_pct`

| | |
|---|---|
| **2025 value** | `2.533` |
| **Availability** | ❌ Not available (estimated) |
| **Method** | D — OLS trend extrapolation (2022–2024) |
| **Source** | Prior dataset values; DCS LFS Annual Reports |
| **Notes** | Annual LFS report for 2025 not yet published. Floor of 1.5% applied. |

---

### `services_employment_share_pct`

| | |
|---|---|
| **2025 value** | `49.9` |
| **Availability** | 🟡 Partial |
| **Method** | B — Quarterly average with interpolation |
| **Source** | DCS LFS Q1 2025 Bulletin (Table 5 & 7); Adaderana Biz, Q3 2025 LFS coverage (January 21, 2026) |
| **Notes** | Q1 = 50.3% (official LFS). Q3 = 49.8% (Adaderana Biz/DCS Q3 bulletin). Q2 interpolated at 49.5%, Q4 at 50.0%. Annual average: 49.9%. |

---

### `youth_lfpr_female_pct`

| | |
|---|---|
| **2025 value** | `17.512` |
| **Availability** | ❌ Not available (estimated) |
| **Method** | D — OLS trend extrapolation (2022–2024) |
| **Source** | DCS LFS Annual Reports; LFS Q1 2025 Table 4 (contextual reference) |
| **Notes** | Annual compiled figure not yet published. LFS Q1 2025 Table 4 shows female LFPR age 20–24 = 36.7% (Q1 snapshot), which is a different metric. Trend extrapolation on 2022–2024 dataset values used as best proxy. Floor of 15.0% applied. |

---

### `remittance_usd`

| | |
|---|---|
| **2025 value** | `8,076,200,000` (≈ $8.08 bn) |
| **Availability** | ✅ Available |
| **Method** | A — Official value |
| **Source** | Central Bank of Sri Lanka (CBSL), Annual External Sector Statistics 2025 |
| **Notes** | CBSL confirmed 2025 annual worker remittances = $8,076.2 million, a record high. 22.8% YoY increase over 2024 ($6,721.6 mn). |

---

### `agri_output_index`

| | |
|---|---|
| **2025 value** | `100.736` |
| **Availability** | ❌ Not available (estimated) |
| **Method** | E — Growth-rate application |
| **Source** | DCS National Accounts 2025 (agriculture sector growth rate); prior dataset values |
| **Notes** | DCS National Accounts confirm agriculture sector grew 1.4% in 2025. Applied to 2024 index value (99.345). Direct index publication not found as of March 2026. |

---

### `crisis_dummy`

| | |
|---|---|
| **2025 value** | `0` |
| **Availability** | ✅ Available |
| **Method** | F — Logical classification |
| **Source** | IMF EFF programme documentation; CBSL Annual Report 2025 |
| **Notes** | 2025 is post-crisis recovery. IMF EFF programme is active but no new debt default or declared economic emergency occurred. Consistent with 2024 classification (also 0). |

---

### `period`

| | |
|---|---|
| **2025 value** | `"recovery"` |
| **Availability** | ✅ Available |
| **Method** | F — Logical classification |
| **Source** | IMF, World Bank Sri Lanka Development Update (April 2025) |
| **Notes** | IMF and World Bank consistently describe 2025 as a recovery phase. World Bank: "growth expected to moderate to 3.5 percent" (later revised upward to 5.0% by DCS). Classification is consistent with 2024. |

---

## Availability Summary

| Column | Availability | Method |
|--------|-------------|--------|
| `gdp_growth_pct` | ✅ Available | Official (DCS) |
| `remittance_usd` | ✅ Available | Official (CBSL) |
| `crisis_dummy` | ✅ Available | Logical |
| `period` | ✅ Available | Logical |
| `unemp_sector_total` | 🟡 Partial | Q1–Q3 official; Q4 estimated |
| `unemp_sector_agri` | 🟡 Partial | Q1 ratio scaled to annual avg |
| `unemp_sector_industry` | 🟡 Partial | Q1 ratio scaled to annual avg |
| `unemp_sector_services` | 🟡 Partial | Q1 ratio scaled to annual avg |
| `tru_female` | 🟡 Partial | Q1 official; Q2–Q4 interpolated |
| `tru_male` | 🟡 Partial | Q1 official; Q2–Q4 interpolated |
| `gdp_usd` | 🟡 Partial | Derived (growth + deflator) |
| `real_gdp_mn_usd` | 🟡 Partial | DCS LKR value converted at avg rate |
| `inflation_pct` | 🟡 Partial | CBSL monthly avg (all 12 months) |
| `services_employment_share_pct` | 🟡 Partial | Q1 + Q3 official; Q2/Q4 interpolated |
| `underemployment_total_pct` | ❌ Not available | OLS trend (2022–2024) |
| `underemployment_male_pct` | ❌ Not available | OLS trend (2022–2024) |
| `underemployment_female_pct` | ❌ Not available | OLS trend (2022–2024) |
| `youth_lfpr_female_pct` | ❌ Not available | OLS trend (2022–2024) |
| `agri_output_index` | ❌ Not available | Agriculture growth rate applied |

---

## Data Sources

| # | Source | Publisher | Type | URL |
|---|--------|-----------|------|-----|
| 1 | LFS Q1 2025 Quarterly Bulletin | Department of Census and Statistics (DCS), Sri Lanka | Official quarterly report | https://www.statistics.gov.lk/Resource/en/LabourForce/Bulletins/LFS_Q1_Bulletin_2025.pdf |
| 2 | LFS Q2 2025 Quarterly Report | Department of Census and Statistics (DCS), Sri Lanka | Official quarterly report | https://www.statistics.gov.lk/Resource/en/LabourForce/Quarterly_Reports/2025Q2report.pdf |
| 3 | LFS Q3 2025 Quarterly Bulletin | Department of Census and Statistics (DCS), Sri Lanka | Official quarterly report | https://www.statistics.gov.lk/ |
| 4 | National Accounts — GDP Q4 and Annual 2025 | Department of Census and Statistics (DCS), Sri Lanka | Official press release | https://www.statistics.gov.lk/ |
| 5 | CCPI Monthly Inflation Press Releases, 2025 | Central Bank of Sri Lanka (CBSL) | Official monthly releases | https://www.cbsl.gov.lk/en/press/press-releases/inflation |
| 6 | CCPI Headline Inflation December 2025 | Central Bank of Sri Lanka (CBSL) | Official press release | https://www.cbsl.gov.lk/en/news/inflation-in-december-2025-ccpi |
| 7 | External Sector Statistics — Remittances 2025 | Central Bank of Sri Lanka (CBSL) | Official annual statistic | https://www.cbsl.gov.lk/ |
| 8 | IMF Country Report No. 25/339 — Sri Lanka | International Monetary Fund | Article IV / EFF review | https://www.imf.org/-/media/files/publications/cr/2025/english/1lkaea2025003-source-pdf.pdf |
| 9 | Sri Lanka Development Update — Staying on Track (April 2025) | World Bank | Bi-annual country report | https://www.worldbank.org/en/country/srilanka/publication/sri-lanka-development-update-2025 |
| 10 | Q3 2025 LFS coverage — "Labour Force Participation Rebounds to 49.9%" | Adaderana Biz English | News report based on DCS bulletin | https://bizenglish.adaderana.lk/sri-lanka-labour-force-participation-rebounds-to-49-9-in-q3-2025-unemployment-holds-steady-at-4-3/ |
| 11 | Q1 2025 LFS coverage — "Unemployment Hits Record Low" | LankaTalks | News report based on DCS bulletin | https://www.lankatalks.com/post/sri-lanka-s-unemployment-hits-record-low-in-q1-2025-amid-economic-recovery |
| 12 | Sri Lanka Inflation Rate (monthly CCPI values) | Trading Economics / CBSL | Data aggregator (CBSL source) | https://tradingeconomics.com/sri-lanka/inflation-cpi |
| 13 | Sri Lanka GDP — Worldometer | Worldometer / IMF | Data aggregator (IMF source) | https://www.worldometers.info/gdp/sri-lanka-gdp/ |

---

## Limitations and Caveats

1. **Q4 2025 LFS not yet released.** As of March 2026, the DCS LFS Q4 2025 bulletin has not been published. All quarterly averages for labour market variables are based on Q1–Q3 data with an estimated Q4.

2. **Annual LFS Report for 2025 not available.** Underemployment by sex, youth LFPR, and annual sectoral breakdown are normally published in the annual LFS report, which is typically released 12–18 months after year-end. The 2025 annual report is not expected before mid-2026.

3. **GDP in USD is an approximation.** Nominal GDP in USD depends on the average annual exchange rate. The CBSL publishes an official average only in the Annual Report (expected mid-2026). The 295 LKR/USD rate used here is an estimate derived from CBSL monthly interbank rates.

4. **`real_gdp_mn_usd` units.** The column appears to represent real GDP in constant 2015 LKR prices (expressed in millions), not constant 2015 USD. The division by the exchange rate maintains comparability with prior year values but does not produce a true constant-price USD figure.

5. **OLS extrapolation assumes linearity.** The trend extrapolation for underemployment and youth LFPR variables assumes a linear continuation of recent trends. Given the structural disruptions of 2022–2023, these extrapolations carry higher uncertainty than the quarterly-based estimates.

6. **Inflation average reflects base effects.** The low annual average inflation figure (0.43%) is heavily influenced by large negative readings in January–March 2025 caused by favourable base effects from the 2022 crisis. This does not reflect deflation; it reflects the statistical comparison against 2022 crisis-era price levels.

---

*Last updated: March 2026. To update this document when Q4 2025 LFS and DCS Annual Report data are published, re-run `estimate_2025.py` with the confirmed values substituted for estimated ones.*
