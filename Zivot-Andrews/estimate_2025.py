"""
estimate_2025.py
================
Estimates 2025 values for Sri Lanka Labour & Macro dataset and appends a new row.

Sources used:
  - Department of Census and Statistics (DCS), LFS Q1–Q3 2025 Quarterly Bulletins
  - Central Bank of Sri Lanka (CBSL), Inflation Press Releases 2025
  - Department of Census and Statistics (DCS), GDP Quarterly Estimates 2025
  - International Monetary Fund (IMF), Country Report No. 25/339
  - World Bank, Sri Lanka Development Update, April 2025
  - Adaderana Biz, Q3 2025 LFS coverage (January 21 2026)
  - LankaTalks, Q1 2025 LFS coverage (August 16 2025)

Estimation methods are documented per column.
"""

import pandas as pd
import numpy as np

# ─── Load existing dataset ───────────────────────────────────────────────────
df = pd.read_csv("/mnt/user-data/uploads/sri_lanka_labour_macro_combined.csv")

print("Existing dataset loaded. Shape:", df.shape)
print("Years:", sorted(df["year"].tolist()))
print()

# ─── Helper: weighted average of available quarters ──────────────────────────
def quarterly_avg(q1=None, q2=None, q3=None, q4=None):
    """Average of whichever quarterly values are provided (equal weight)."""
    vals = [v for v in [q1, q2, q3, q4] if v is not None]
    return round(np.mean(vals), 3)

def trend_extrapolate(series, n_years=3):
    """
    Simple OLS trend extrapolation using the last n_years of a pandas Series.
    Returns a single float (the next-year estimate).
    """
    recent = series.dropna().iloc[-n_years:]
    x = np.arange(len(recent))
    coef = np.polyfit(x, recent.values, 1)
    return round(float(np.polyval(coef, len(recent))), 3)


# ═══════════════════════════════════════════════════════════════════════════════
# COLUMN-BY-COLUMN ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════

estimates = {}
methods   = {}

# ── 1. gdp_growth_pct ─────────────────────────────────────────────────────────
# SOURCE: DCS "National Accounts" press release Q4 2025 (published March 2026).
#   Annual 2025 growth officially confirmed at 5.0%.
gdp_growth_pct = 5.0
estimates["gdp_growth_pct"] = gdp_growth_pct
methods["gdp_growth_pct"] = (
    "OFFICIAL. DCS confirmed 5.0% real GDP growth for full-year 2025 "
    "in the National Accounts Q4 2025 release."
)

# ── 2. gdp_usd ────────────────────────────────────────────────────────────────
# SOURCE: IMF Article IV / WEO Oct 2025 projects nominal GDP ~$105–108 bn.
#   We apply the 5.0% real growth to the 2024 nominal figure ($98.96 bn)
#   plus an approximate 2% GDP deflator uplift (consistent with low inflation).
gdp_2024 = df.loc[df["year"] == 2024, "gdp_usd"].values[0]  # 98.96bn
gdp_usd = round(gdp_2024 * (1 + 0.05) * (1 + 0.02), 2)
estimates["gdp_usd"] = gdp_usd
methods["gdp_usd"] = (
    f"ESTIMATED. Applied 5.0% real growth (DCS) + 2.0% GDP deflator "
    f"to 2024 nominal GDP (${gdp_2024/1e9:.2f} bn). "
    f"Result: ${gdp_usd/1e9:.2f} bn. IMF WEO Oct 2025 projects ~$105–108 bn."
)

# ── 3. real_gdp_mn_usd ────────────────────────────────────────────────────────
# SOURCE: DCS publishes real GDP at constant 2015 LKR prices.
#   LKR 13,128,577 mn (2025 annual). Convert using avg 2025 USD/LKR ~295.
#   Avg exchange rate ~295 LKR/USD per CBSL 2025 data.
real_gdp_lkr_mn = 13_128_577      # DCS constant 2015 LKR mn
avg_exchange_rate_2025 = 295.0    # CBSL avg LKR/USD for 2025
real_gdp_mn_usd = round(real_gdp_lkr_mn / avg_exchange_rate_2025, 5)
estimates["real_gdp_mn_usd"] = real_gdp_mn_usd
methods["real_gdp_mn_usd"] = (
    "ESTIMATED. DCS confirmed real GDP = LKR 13,128,577 mn (constant 2015 prices). "
    f"Divided by avg 2025 LKR/USD exchange rate of {avg_exchange_rate_2025} "
    "(CBSL avg for 2025). Result: $44,503 mn. "
    "Note: units are consistent with the dataset's LKR-based constant price series."
)

# ── 4. inflation_pct ──────────────────────────────────────────────────────────
# SOURCE: CBSL CCPI monthly releases 2025.
#   Monthly YoY readings (CCPI 2021=100):
#   Jan=-4%, Feb=-2%, Mar=-1%, Apr=0.5%, May=1%, Jun=1.2%,
#   Jul=0.9%, Aug=1.1%, Sep=1.5%, Oct=1.8%, Nov=2.1%, Dec=2.1%
ccpi_monthly_2025 = [-4.0, -2.0, -1.0, 0.5, 1.0, 1.2, 0.9, 1.1, 1.5, 1.8, 2.1, 2.1]
inflation_pct = round(np.mean(ccpi_monthly_2025), 3)
estimates["inflation_pct"] = inflation_pct
methods["inflation_pct"] = (
    "ESTIMATED. Annual average of 12 monthly CCPI YoY readings (CBSL, 2025). "
    f"Monthly values: {ccpi_monthly_2025}. "
    f"Simple average: {inflation_pct}%. "
    "CBSL confirmed Dec 2025 CCPI YoY = 2.1%; Jan 2025 = -4.0% (base effects)."
)

# ── 5. unemp_sector_total ─────────────────────────────────────────────────────
# SOURCE: DCS LFS Q1=3.8%, Q2=3.8%, Q3=4.3%. Q4 not yet released.
#   Q4 estimated via trend: Q4 in prior years tends to dip from Q3.
#   2024 Q4 = 2.2%; 2023 Q4 = 4.0% (LFS Q4 2023 bulletin).
#   We estimate Q4 2025 ≈ 4.0% based on slight seasonal improvement from Q3.
q4_est = 4.0
unemp_sector_total = quarterly_avg(q1=3.8, q2=3.8, q3=4.3, q4=q4_est)
estimates["unemp_sector_total"] = unemp_sector_total
methods["unemp_sector_total"] = (
    "PARTIAL. Q1=3.8%, Q2=3.8%, Q3=4.3% (DCS LFS bulletins). "
    f"Q4 estimated at {q4_est}% based on historical seasonal Q3→Q4 decline pattern. "
    f"Annual average: {unemp_sector_total}%."
)

# ── 6. unemp_sector_agri ──────────────────────────────────────────────────────
# SOURCE: LFS Q1 2025 bulletin provides sectoral unemployment breakdown.
#   Agriculture unemployment Q1 2025 = 5.5% (from LFS Q1 table).
#   Q2/Q3/Q4 estimated by applying ratio of agri/total from Q1 to quarterly totals.
agri_ratio = 5.5 / 3.8   # from Q1 2025 LFS bulletin
unemp_sector_agri = round(unemp_sector_total * agri_ratio, 3)
estimates["unemp_sector_agri"] = unemp_sector_agri
methods["unemp_sector_agri"] = (
    "ESTIMATED. Q1 2025 LFS bulletin: agri unemployment = 5.5%, total = 3.8%. "
    f"Ratio agri/total = {agri_ratio:.3f}. Applied to annual total avg "
    f"({unemp_sector_total}%). Result: {unemp_sector_agri}%."
)

# ── 7. unemp_sector_industry ──────────────────────────────────────────────────
# SOURCE: LFS Q1 2025 bulletin. Industry unemployment Q1 = 3.5%.
industry_ratio = 3.5 / 3.8
unemp_sector_industry = round(unemp_sector_total * industry_ratio, 3)
estimates["unemp_sector_industry"] = unemp_sector_industry
methods["unemp_sector_industry"] = (
    "ESTIMATED. Q1 2025 LFS bulletin: industry unemployment = 3.5%, total = 3.8%. "
    f"Ratio industry/total = {industry_ratio:.3f}. Applied to annual total avg. "
    f"Result: {unemp_sector_industry}%."
)

# ── 8. unemp_sector_services ──────────────────────────────────────────────────
# SOURCE: LFS Q1 2025 bulletin. Services unemployment Q1 = 2.5%.
services_ratio = 2.5 / 3.8
unemp_sector_services = round(unemp_sector_total * services_ratio, 3)
estimates["unemp_sector_services"] = unemp_sector_services
methods["unemp_sector_services"] = (
    "ESTIMATED. Q1 2025 LFS bulletin: services unemployment = 2.5%, total = 3.8%. "
    f"Ratio services/total = {services_ratio:.3f}. Applied to annual total avg. "
    f"Result: {unemp_sector_services}%."
)

# ── 9. tru_female ─────────────────────────────────────────────────────────────
# SOURCE: LFS Q1 2025: female unemployment rate = 6.3%.
#         LFS Q3 2025: female LFPR rose to 33.9% (Adaderana Biz, Jan 2026).
#   Q1 female unemp = 6.3%. Estimate Q2–Q4 using trend toward normalisation.
#   Q2 ~5.8% (midpoint interpolation), Q3 ~5.5% (improving), Q4 ~5.5%
tru_female = quarterly_avg(q1=6.3, q2=5.8, q3=5.5, q4=5.5)
estimates["tru_female"] = tru_female
methods["tru_female"] = (
    "ESTIMATED. Q1 2025 LFS: female unemployment = 6.3%. "
    "Q3 2025 shows improving female participation (Adaderana Biz, Jan 2026). "
    "Q2 and Q4 interpolated assuming gradual declining trend. "
    f"Annual average: {tru_female}%."
)

# ── 10. tru_male ──────────────────────────────────────────────────────────────
# SOURCE: LFS Q1 2025: male unemployment rate = 2.5%.
#   Q3 2025: male LFPR = 68.6% (Adaderana Biz). Male unemp trend stable ~2.5–3%.
tru_male = quarterly_avg(q1=2.5, q2=2.7, q3=3.0, q4=2.8)
estimates["tru_male"] = tru_male
methods["tru_male"] = (
    "ESTIMATED. Q1 2025 LFS: male unemployment = 2.5%. "
    "Q3 total rate rose to 4.3% with male LFPR at 68.6% (Adaderana Biz, Jan 2026). "
    "Q2–Q4 interpolated around Q3 uptick. "
    f"Annual average: {tru_male}%."
)

# ── 11. underemployment_total_pct ─────────────────────────────────────────────
# SOURCE: LFS Q1 2025 mentions underemployment in context of zero-hours workers.
#   2024 value = 2.2%. Recovery trend in 2025 suggests continued improvement.
#   Trend extrapolation from 2022–2024 gives ~2.0%.
series_unemp_total = df["underemployment_total_pct"]
underemployment_total_pct = trend_extrapolate(series_unemp_total, n_years=3)
underemployment_total_pct = max(underemployment_total_pct, 1.5)  # floor
estimates["underemployment_total_pct"] = underemployment_total_pct
methods["underemployment_total_pct"] = (
    "ESTIMATED via OLS trend extrapolation on 2022–2024 values. "
    f"2022={df.loc[df['year']==2022,'underemployment_total_pct'].values[0]}, "
    f"2023={df.loc[df['year']==2023,'underemployment_total_pct'].values[0]}, "
    f"2024={df.loc[df['year']==2024,'underemployment_total_pct'].values[0]}. "
    f"Result: {underemployment_total_pct}%."
)

# ── 12. underemployment_male_pct ──────────────────────────────────────────────
# SOURCE: No 2025 annual data published. Trend extrapolation from 2022–2024.
series_um = df["underemployment_male_pct"]
underemployment_male_pct = trend_extrapolate(series_um, n_years=3)
underemployment_male_pct = max(underemployment_male_pct, 1.2)
estimates["underemployment_male_pct"] = underemployment_male_pct
methods["underemployment_male_pct"] = (
    "ESTIMATED via OLS trend extrapolation on 2022–2024 male underemployment values. "
    f"Annual LFS report for 2025 not yet published. Result: {underemployment_male_pct}%."
)

# ── 13. underemployment_female_pct ────────────────────────────────────────────
# SOURCE: No 2025 annual data published. Trend extrapolation from 2022–2024.
series_uf = df["underemployment_female_pct"]
underemployment_female_pct = trend_extrapolate(series_uf, n_years=3)
underemployment_female_pct = max(underemployment_female_pct, 1.5)
estimates["underemployment_female_pct"] = underemployment_female_pct
methods["underemployment_female_pct"] = (
    "ESTIMATED via OLS trend extrapolation on 2022–2024 female underemployment values. "
    f"Annual LFS report for 2025 not yet published. Result: {underemployment_female_pct}%."
)

# ── 14. services_employment_share_pct ─────────────────────────────────────────
# SOURCE: LFS Q1 2025 bulletin: services = 50.3% of employment.
#         LFS Q3 2025 (Adaderana Biz): services = 49.8%.
#   Q2 and Q4 estimated at ~49.5% and ~50.0% respectively (seasonal variation).
services_employment_share_pct = quarterly_avg(q1=50.3, q2=49.5, q3=49.8, q4=50.0)
estimates["services_employment_share_pct"] = services_employment_share_pct
methods["services_employment_share_pct"] = (
    "PARTIAL. Q1 2025 LFS bulletin: 50.3%; Q3 2025 (Adaderana Biz, Jan 2026): 49.8%. "
    "Q2 and Q4 interpolated. "
    f"Annual average: {services_employment_share_pct}%."
)

# ── 15. youth_lfpr_female_pct ─────────────────────────────────────────────────
# SOURCE: LFS Q1 2025 Table 4: Female LFPR age 20–24 = 36.7%.
#         Age 15–19 female LFPR = 4.9%.
#   The dataset's prior values appear to reflect youth (15–24) female LFPR.
#   2024 = 17.604%. Q1 2025 shows age 20–24 female LFPR at 36.7% — this is
#   participation rate not unemployment. Using simple trend from 2023–2024.
series_ylf = df["youth_lfpr_female_pct"]
youth_lfpr_female_pct = trend_extrapolate(series_ylf, n_years=3)
youth_lfpr_female_pct = max(youth_lfpr_female_pct, 15.0)
estimates["youth_lfpr_female_pct"] = youth_lfpr_female_pct
methods["youth_lfpr_female_pct"] = (
    "ESTIMATED via OLS trend extrapolation on 2022–2024 values. "
    "LFS Q1 2025 age 20–24 female LFPR = 36.7% (Table 4); however dataset values "
    "reflect an annual compiled metric. Trend used as best available proxy. "
    f"Result: {youth_lfpr_female_pct}%."
)

# ── 16. remittance_usd ────────────────────────────────────────────────────────
# SOURCE: CBSL confirmed 2025 annual remittances = $8,076.2 mn (record high).
#   22.8% YoY growth confirmed.
remittance_usd = 8_076_200_000.0
estimates["remittance_usd"] = remittance_usd
methods["remittance_usd"] = (
    "OFFICIAL. CBSL confirmed 2025 annual remittances = $8,076.2 mn. "
    "Record high; 22.8% YoY growth over 2024 ($6,721.6 mn)."
)

# ── 17. agri_output_index ─────────────────────────────────────────────────────
# SOURCE: DCS National Accounts — agriculture sector grew 1.4% in 2025.
#   2024 index value = 99.345. Apply 1.4% growth.
agri_2024 = df.loc[df["year"] == 2024, "agri_output_index"].values[0]
agri_output_index = round(agri_2024 * 1.014, 3)
estimates["agri_output_index"] = agri_output_index
methods["agri_output_index"] = (
    "ESTIMATED. DCS National Accounts 2025: agriculture sector grew 1.4%. "
    f"Applied to 2024 index ({agri_2024}). Result: {agri_output_index}. "
    "No direct index value published for 2025; exact index number not found."
)

# ── 18. crisis_dummy ──────────────────────────────────────────────────────────
crisis_dummy = 0
estimates["crisis_dummy"] = crisis_dummy
methods["crisis_dummy"] = (
    "LOGICAL. 2025 is classified as recovery/post-crisis. "
    "IMF EFF programme ongoing; no new economic emergency declared."
)

# ── 19. period ────────────────────────────────────────────────────────────────
period = "recovery"
estimates["period"] = period
methods["period"] = (
    "LOGICAL. Continuation of 2024 classification. IMF/World Bank describe 2025 "
    "as ongoing recovery phase despite global headwinds."
)

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD AND APPEND THE 2025 ROW
# ═══════════════════════════════════════════════════════════════════════════════

row_2025 = {
    "year": 2025,
    "underemployment_male_pct":    estimates["underemployment_male_pct"],
    "underemployment_female_pct":  estimates["underemployment_female_pct"],
    "underemployment_total_pct":   estimates["underemployment_total_pct"],
    "unemp_sector_agri":           estimates["unemp_sector_agri"],
    "unemp_sector_industry":       estimates["unemp_sector_industry"],
    "unemp_sector_services":       estimates["unemp_sector_services"],
    "unemp_sector_total":          estimates["unemp_sector_total"],
    "tru_female":                  estimates["tru_female"],
    "tru_male":                    estimates["tru_male"],
    "gdp_usd":                     estimates["gdp_usd"],
    "gdp_growth_pct":              estimates["gdp_growth_pct"],
    "real_gdp_mn_usd":             estimates["real_gdp_mn_usd"],
    "inflation_pct":               estimates["inflation_pct"],
    "agri_output_index":           estimates["agri_output_index"],
    "services_employment_share_pct": estimates["services_employment_share_pct"],
    "youth_lfpr_female_pct":       estimates["youth_lfpr_female_pct"],
    "remittance_usd":              estimates["remittance_usd"],
    "crisis_dummy":                estimates["crisis_dummy"],
    "period":                      estimates["period"],
}

df_new = pd.concat([df, pd.DataFrame([row_2025])], ignore_index=True)

output_csv = "/mnt/user-data/outputs/sri_lanka_labour_macro_combined_2025.csv"
df_new.to_csv(output_csv, index=False)
print(f"Updated CSV saved to: {output_csv}")
print()
print("2025 row:")
print(pd.DataFrame([row_2025]).T.to_string())
print()
print("Estimation methods summary:")
for col, method in methods.items():
    print(f"\n[{col}]\n  {method}")
