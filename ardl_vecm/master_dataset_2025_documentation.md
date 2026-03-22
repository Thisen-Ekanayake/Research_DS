# 2025 Data Estimation — Sri Lanka Master Dataset

**Document:** Methodology, Data Sources, and Availability  
**Dataset:** `master_dataset_2025.csv`  
**Script:** `estimate_master_2025.py`  
**Prepared:** March 2026  
**Total columns:** 92 across 3 groups  
**Coverage:** Year 2025 row appended to 2015–2024 dataset

---

## 1. Overview

The original dataset spans 2015–2024 across 92 columns organised into three groups:

| Group | Columns | Description |
|-------|---------|-------------|
| Labour / Macro | 16 | Unemployment, underemployment, informality, youth labour, GDP, inflation, exchange rate |
| Agricultural Production Indices | 72 | FAO-style production indices for individual crops, livestock, and aggregate categories |
| Remittances | 4 | World Bank / CBSL BoP remittance flows (received, paid, % of GDP, personal transfers) |

---

## 2. Estimation Tiers

| Tier | Label | Description |
|------|-------|-------------|
| A | **Official** | Confirmed in a published government or international source |
| B | **Partial** | Derived from 2–3 quarterly values; Q4 2025 not yet published |
| C | **Ratio** | Sub-indicator ratio from Q1 2025 LFS bulletin scaled to annual total |
| D | **OLS trend** | Linear trend extrapolated from last 3 (or 5 for high-volatility) years |
| E | **Growth rate** | Known sector growth rate applied to prior year value |
| F | **Logical** | Non-numeric or categorical, assigned from economic context |

---

## 3. Group 1 — Labour and Macro Indicators

### `Underemployment_Rate`

| | |
|---|---|
| **2025 value** | `2.5` |
| **Availability** | ❌ Not available |
| **Tier** | D — OLS trend on 2022–2024 (2.3, 2.5, 2.4) |
| **Source** | DCS LFS Annual Reports (2025 annual not yet published) |
| **Notes** | Floor of 1.5% applied. 2023 spike was crisis-related; 2024 normalisation continued. |

---

### `Underemployment_Male`

| | |
|---|---|
| **2025 value** | `2.2` |
| **Availability** | ❌ Not available |
| **Tier** | D — OLS trend on 2022–2024 |
| **Source** | DCS LFS Annual Reports |
| **Notes** | Floor of 1.2% applied. |

---

### `Underemployment_Female`

| | |
|---|---|
| **2025 value** | `3.167` |
| **Availability** | ❌ Not available |
| **Tier** | D — OLS trend on 2022–2024 (2.8, 3.1, 3.0) |
| **Source** | DCS LFS Annual Reports |
| **Notes** | Floor of 1.5% applied. Female underemployment structurally higher than male. |

---

### `Real_GDP`

| | |
|---|---|
| **2025 value** | `43,642.63` |
| **Availability** | 🟡 Partial |
| **Tier** | E — DCS LKR constant-price GDP converted at 2025 avg exchange rate |
| **Source** | DCS National Accounts Annual 2025; exchange-rates.org USD/LKR 2025 |
| **Notes** | DCS confirmed 2025 real GDP = LKR 13,128,577 mn (constant 2015 prices). Divided by 2025 annual average USD/LKR = 300.82. Value is consistent with the dataset's LKR-to-USD constant-price conversion convention. |

---

### `GDP_Growth_Rate`

| | |
|---|---|
| **2025 value** | `5.0` |
| **Availability** | ✅ Available |
| **Tier** | A — Official |
| **Source** | Department of Census and Statistics (DCS), National Accounts Annual 2025 |
| **Notes** | Q4 2025 growth = 4.8%; annual 2025 = 5.0%. Confirmed in DCS press release (published March 2026). |

---

### `Inflation_Rate`

| | |
|---|---|
| **2025 value** | `0.433` |
| **Availability** | 🟡 Partial |
| **Tier** | B — Simple average of 12 monthly CCPI YoY readings |
| **Source** | Central Bank of Sri Lanka (CBSL), monthly CCPI press releases Jan–Dec 2025 |
| **Notes** | Monthly readings: Jan −4.0%, Feb −2.0%, Mar −1.0%, Apr +0.5%, May +1.0%, Jun +1.2%, Jul +0.9%, Aug +1.1%, Sep +1.5%, Oct +1.8%, Nov +2.1%, Dec +2.1%. Low annual average reflects strong base effects from 2022 crisis-year prices; not indicative of deflation. |

---

### `Exchange_Rate_LKR_USD`

| | |
|---|---|
| **2025 value** | `300.82` |
| **Availability** | ✅ Available |
| **Tier** | A — Official annual average |
| **Source** | exchange-rates.org (sourced from ECB/market data), USD/LKR 2025 annual average; corroborated by FRED EXSLUS (Nov 2025 = 306.14) |
| **Notes** | Range during 2025: 293.07 (Mar 30) to 310.15 (Dec 29). Annual average = 300.82 LKR/USD. |

---

### `Informal_Pct`

| | |
|---|---|
| **2025 value** | `56.93` |
| **Availability** | ❌ Not available |
| **Tier** | D — OLS trend on 2022–2024 (57.4, 58.0, 56.9) |
| **Source** | DCS LFS Annual Reports; ILO ILOSTAT |
| **Notes** | Floor of 50% applied. Declining trend consistent with post-crisis formalisation. Annual LFS 2025 not yet published. |

---

### `Informal_Male_Pct`

| | |
|---|---|
| **2025 value** | `60.8` (approx) |
| **Availability** | ❌ Not available |
| **Tier** | D — OLS trend on 2022–2024 |
| **Source** | DCS LFS Annual Reports |
| **Notes** | Floor of 55% applied. |

---

### `Informal_Female_Pct`

| | |
|---|---|
| **2025 value** | `47.0` (approx) |
| **Availability** | ❌ Not available |
| **Tier** | D — OLS trend on 2022–2024 (48.6, 49.5, 47.7) |
| **Source** | DCS LFS Annual Reports |
| **Notes** | Floor of 42% applied. Female informal share declining. |

---

### `Youth_LFPR_15_24`

| | |
|---|---|
| **2025 value** | `22.87` |
| **Availability** | ❌ Not available |
| **Tier** | D — OLS trend on 2022–2024 (25.3, 24.0, 23.8) |
| **Source** | DCS LFS Annual Reports; contextual reference from LFS Q1 2025 Table 4 |
| **Notes** | LFS Q1 2025 Table 4: LFPR for 15–19 = 7.5%, 20–24 = 48.4%. The dataset's annual composite figure is not yet published. OLS trend on prior years used. Floor 20% applied. |

---

### `Youth_Unemployment_15_24`

| | |
|---|---|
| **2025 value** | `20.43` |
| **Availability** | 🟡 Partial |
| **Tier** | B — Q1 official + Q2–Q4 interpolated |
| **Source** | DCS LFS Q1 2025 Bulletin Table 8 |
| **Notes** | Q1 = 19.7% (official). Q3 estimated ~21.0% (seasonal rise). Q2 = 20.5%, Q4 = 20.5% (interpolated). Annual average = 20.43%. |

---

### `Youth_Unemployment_Male`

| | |
|---|---|
| **2025 value** | `15.5` |
| **Availability** | 🟡 Partial |
| **Tier** | B — Q1 official + interpolated |
| **Source** | DCS LFS Q1 2025 Bulletin Table 8 |
| **Notes** | Q1 = 15.0% (official). Q2–Q4 interpolated around slight seasonal uptick in Q3. |

---

### `Youth_Unemployment_Female`

| | |
|---|---|
| **2025 value** | `26.58` |
| **Availability** | 🟡 Partial |
| **Tier** | B — Q1 official + interpolated |
| **Source** | DCS LFS Q1 2025 Bulletin Table 8 |
| **Notes** | Q1 = 27.3% (official). Gradual declining trend assumed for Q2–Q4 (26.5%, 26.0%, 26.5%). |

---

### `Unemployment_Rate`

| | |
|---|---|
| **2025 value** | `3.975` |
| **Availability** | 🟡 Partial |
| **Tier** | B — Q1–Q3 official; Q4 estimated |
| **Source** | DCS LFS Q1 2025 Bulletin; DCS LFS Q3 2025 Summary; DCS website |
| **Notes** | Q1 = 3.8%, Q2 = 3.8%, Q3 = 4.3% (all official). Q4 estimated at 4.0% based on seasonal Q3→Q4 decline in prior years (e.g. 2023: Q3 = 3.7%, Q4 = 4.0%). Q4 bulletin not released as of March 2026. |

---

## 4. Group 2 — Agricultural Production Indices (72 columns)

### General notes

All 72 agricultural production index columns follow FAO methodology (base year 2014–2016 = 100). FAOSTAT typically releases country-level data for year *T* in *T+18 to T+24 months*. **2025 FAOSTAT data for Sri Lanka is not yet available.** The dataset's 2024 values appear to be sourced from preliminary DCS national production data or early FAO estimates.

**Estimation strategy:**

- **Aggregate indices** (Agriculture, Food, Non-Food, Crops, Livestock, aggregate commodity groups): applied DCS-confirmed 2025 agriculture sector growth of **+1.4%** as a baseline, adjusted upward or downward based on available sector-specific 2025 data.
- **Commodity-level indices**: OLS linear trend on the last 3 years. For series with a coefficient of variation (CV) above 0.5 across the last 3 years (indicating high volatility, e.g. from crisis disruption), a 5-year trend is used for stability.
- The `Eggs_from_other_birds_in_shell_fresh` index is **constant at 100.0** across all 10 years and is maintained at 100.0 for 2025.

### Key sector-specific anchors (2025)

| Commodity | 2025 anchor | Source |
|-----------|-------------|--------|
| Agriculture (total) | +1.4% sector growth | DCS National Accounts 2025 |
| Tea | −5.0% (production decline) | CBSL / Forbes & Walker 2025 |
| Natural rubber | −2.0% (slight decline) | CBSL 2025 |
| Rice (paddy) | +2.0% (post-fertiliser normalisation) | DCS / CBSL 2025 |
| Coconuts | +2.0% (recovery) | DCS 2025 |
| Chickens (meat) | +3.5% | DCS livestock estimates |
| Hen eggs | +2.5% | DCS livestock estimates |
| Total milk | +2.0% | DCS livestock estimates |
| Cereals | +2.0% | DCS 2025 |
| Livestock (index) | +3.0% | DCS 2025 |
| Vegetables & Fruit | +1.5% | DCS 2025 |

### Selected commodity notes

**`AgriProdIdx_Tea_leaves`** — Sri Lanka's tea production has faced multiple headwinds in 2025: ongoing fertiliser costs, adverse weather in key growing regions, and export market competition. CBSL and Forbes & Walker 2025 reports indicate a ~5% production decline year-on-year. Applied to 2024 index value of 88.37, yielding ~83.95.

**`AgriProdIdx_Natural_rubber_in_primary_forms`** — Rubber production continued a long-term structural decline. A −2.0% rate was applied to the 2024 index (74.43), yielding ~72.94.

**`AgriProdIdx_Nutmeg_mace_cardamoms_raw`** — This index has shown extreme values (566.53 in 2024 vs. 36.78 in 2023), indicating high volatility. The 3-year OLS trend is used despite the instability; results should be treated with caution.

**`AgriProdIdx_Mustard_seed`** — CV = 0.91 over last 3 years; 5-year OLS trend applied.

**`AgriProdIdx_Soya_beans`** — CV = 0.33; borderline volatile; 3-year trend used with floor of 10.

---

## 5. Group 3 — Remittances

### `Remit_Personal_remittances_received_current_US$`

| | |
|---|---|
| **2025 value** | `8,076,200,000` |
| **Availability** | ✅ Available |
| **Tier** | A — Official |
| **Source** | Central Bank of Sri Lanka (CBSL), Annual External Sector Statistics 2025 |
| **Notes** | Record high. 22.8% YoY growth over 2024 ($6,721.6 mn). Monthly data confirms: Dec 2025 = $879.1 mn (highest single month). |

---

### `Remit_Personal_remittances_received_pct_of_GDP`

| | |
|---|---|
| **2025 value** | `7.62` |
| **Availability** | 🟡 Partial |
| **Tier** | E — Remittances received ÷ estimated nominal GDP |
| **Source** | CBSL (numerator); DCS + IMF WEO (GDP denominator) |
| **Notes** | Denominator = estimated 2025 nominal GDP of ~$105.99 bn (5.0% real growth + 2.0% deflator applied to 2024 nominal). Official GDP-in-USD figure not yet published. |

---

### `Remit_Personal_transfers_receipts_BoP_current_US$`

| | |
|---|---|
| **2025 value** | `~7,900,495,815` |
| **Availability** | 🟡 Partial |
| **Tier** | E — 2024 BoP/received ratio applied to 2025 received total |
| **Source** | CBSL Balance of Payments data; 2024 ratio = 0.97824 |
| **Notes** | Personal transfers (BoP) excludes compensation of employees and historically runs ~2% below total received. 2025 BoP data not yet published separately. |

---

### `Remit_Personal_remittances_paid_current_US$`

| | |
|---|---|
| **2025 value** | `~100,000,000` (floor) |
| **Availability** | ❌ Not available |
| **Tier** | D — OLS trend on 2022–2024 paid outflows |
| **Source** | CBSL / World Bank BoP data 2022–2024 |
| **Notes** | 2022 = $364.9 mn, 2023 = $185.2 mn, 2024 = $166.1 mn — declining trend. OLS projects further decline; floor of $100 mn applied. 2025 outflow data not yet published. |

---

## 6. Availability Summary

### Labour / Macro

| Column | Availability | Tier |
|--------|-------------|------|
| `GDP_Growth_Rate` | ✅ Available | A — DCS official |
| `Exchange_Rate_LKR_USD` | ✅ Available | A — exchange-rates.org / CBSL |
| `Unemployment_Rate` | 🟡 Partial | B — Q1–Q3 official; Q4 estimated |
| `Youth_Unemployment_15_24` | 🟡 Partial | B — Q1 official; Q2–Q4 interpolated |
| `Youth_Unemployment_Male` | 🟡 Partial | B — Q1 official; Q2–Q4 interpolated |
| `Youth_Unemployment_Female` | 🟡 Partial | B — Q1 official; Q2–Q4 interpolated |
| `Inflation_Rate` | 🟡 Partial | B — 12-month CCPI average (all months confirmed) |
| `Real_GDP` | 🟡 Partial | E — DCS LKR GDP ÷ avg exchange rate |
| `Underemployment_Rate` | ❌ Not available | D — OLS trend |
| `Underemployment_Male` | ❌ Not available | D — OLS trend |
| `Underemployment_Female` | ❌ Not available | D — OLS trend |
| `Informal_Pct` | ❌ Not available | D — OLS trend |
| `Informal_Male_Pct` | ❌ Not available | D — OLS trend |
| `Informal_Female_Pct` | ❌ Not available | D — OLS trend |
| `Youth_LFPR_15_24` | ❌ Not available | D — OLS trend |

### Agricultural Production Indices

| Column | Availability | Tier |
|--------|-------------|------|
| All 72 AgriProdIdx columns | ❌ Not available (FAOSTAT 2025 not released) | D or E |
| `AgriProdIdx_Eggs_from_other_birds_*` | Maintained constant | E — historical constant |

### Remittances

| Column | Availability | Tier |
|--------|-------------|------|
| `Remit_Personal_remittances_received_current_US$` | ✅ Available | A — CBSL official |
| `Remit_Personal_remittances_received_pct_of_GDP` | 🟡 Partial | E — derived from official received + estimated GDP |
| `Remit_Personal_transfers_receipts_BoP_current_US$` | 🟡 Partial | E — ratio from 2024 BoP relationship |
| `Remit_Personal_remittances_paid_current_US$` | ❌ Not available | D — OLS trend |

---

## 7. Data Sources

| # | Source | Publisher | URL |
|---|--------|-----------|-----|
| 1 | LFS Q1 2025 Quarterly Bulletin | DCS Sri Lanka | https://www.statistics.gov.lk/Resource/en/LabourForce/Bulletins/LFS_Q1_Bulletin_2025.pdf |
| 2 | LFS Q3 2025 Summary Report | DCS Sri Lanka | https://www.statistics.gov.lk/ |
| 3 | National Accounts Annual 2025 (GDP Q4 & Full Year) | DCS Sri Lanka | https://www.statistics.gov.lk/ |
| 4 | CCPI Monthly Press Releases 2025 (Jan–Dec) | CBSL | https://www.cbsl.gov.lk/en/press/press-releases/inflation |
| 5 | Annual External Sector Statistics 2025 — Remittances | CBSL | https://www.cbsl.gov.lk/ |
| 6 | USD/LKR Exchange Rate History 2025 | exchange-rates.org | https://www.exchange-rates.org/exchange-rate-history/usd-lkr-2025 |
| 7 | USD/LKR Spot Exchange Rate (EXSLUS) — monthly | FRED / Federal Reserve | https://fred.stlouisfed.org/series/EXSLUS |
| 8 | IMF Country Report No. 25/339 — Sri Lanka Article IV / EFF Review | IMF | https://www.imf.org/-/media/files/publications/cr/2025/english/1lkaea2025003-source-pdf.pdf |
| 9 | Sri Lanka Development Update — Staying on Track (April 2025) | World Bank | https://www.worldbank.org/en/country/srilanka/publication/sri-lanka-development-update-2025 |
| 10 | Q3 2025 LFS coverage — "Labour Force Participation Rebounds to 49.9%" | Adaderana Biz English | https://bizenglish.adaderana.lk/sri-lanka-labour-force-participation-rebounds-to-49-9-in-q3-2025-unemployment-holds-steady-at-4-3/ |
| 11 | FAOSTAT Production Indices — Sri Lanka | FAO | https://www.fao.org/faostat/en/#data/QI |
| 12 | Sri Lanka Remittances monthly (CBSL source) | Trading Economics | https://tradingeconomics.com/sri-lanka/remittances |
| 13 | Tea Production 2025 — Sri Lanka Tea Board / CBSL | Forbes & Walker / CBSL Annual Report | https://www.cbsl.gov.lk/ |
| 14 | ILO Modelled Estimates — Informal Employment | ILOSTAT | https://ilostat.ilo.org/data/ |

---

## 8. Limitations and Caveats

1. **Q4 2025 LFS not yet released.** As of March 2026, the DCS Labour Force Survey Q4 2025 bulletin has not been published. All quarterly-averaged labour market figures incorporate an estimated Q4 value based on historical seasonal patterns.

2. **Annual LFS Report 2025 not available.** Underemployment by sex, informal employment by sex, and annual youth LFPR are published in the annual LFS report, typically released 12–18 months after year-end. The 2025 annual report is not expected before mid-to-late 2026.

3. **FAOSTAT 2025 not released.** All 72 agricultural production index estimates are model-derived (OLS trend or growth-rate application). FAO typically publishes country indices for year T in months 18–24 of T+1. When the 2025 release becomes available, all AgriProdIdx estimates should be replaced with official values.

4. **Tea and rubber are high-uncertainty estimates.** The −5% tea estimate is based on CBSL and industry commentary; the exact official production figure will be confirmed in the CBSL Annual Report 2025 (expected Q2 2026). Rubber similarly lacks a confirmed 2025 production figure.

5. **Nutmeg/cardamom index is inherently unreliable.** The `AgriProdIdx_Nutmeg_mace_cardamoms_raw` column shows values of 566.53 in 2024 and 36.78 in 2023 — an extreme swing unlikely to reflect underlying production reality. The OLS trend on this series should be treated with caution; users are advised to verify with FAOSTAT when available.

6. **Real_GDP column convention.** The dataset appears to store real GDP in LKR constant 2015 prices divided by a contemporaneous exchange rate, not in true constant-price USD. This is consistent with prior years' values. The 2025 estimate follows this convention using avg USD/LKR = 300.82.

7. **Nominal GDP (for % of GDP calculations).** There is no official 2025 nominal GDP in USD. The denominator for `Remit_Personal_remittances_received_pct_of_GDP` uses a derived estimate ($105.99 bn) and will be superseded once DCS or the IMF publishes official nominal GDP.

8. **OLS trend linearity assumption.** The OLS extrapolations assume a continuation of recent linear trends. Post-crisis recovery paths are rarely linear, and structural breaks (policy changes, weather, external shocks) can make these projections unreliable for individual commodity indices. For aggregate indices (Food, Agriculture, Crops), the sector growth anchor provides a more reliable constraint.

---

*Last updated: March 2026. To update when Q4 2025 LFS data, FAOSTAT 2025 release, or CBSL Annual Report 2025 becomes available, replace the relevant estimated values in `estimate_master_2025.py` with confirmed figures and re-run.*
