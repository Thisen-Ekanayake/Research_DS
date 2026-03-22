"""
EconomicDataloader.py
=====================
Loads and exposes all economic and labour variables from the master dataset
and individual source CSVs. Updated to include remittances, agricultural
output index, informal employment, part-time employment, discouraged workers,
and real exchange rate (all previously missing or KNN-imputed).
"""

import pandas as pd
import numpy as np
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
ECONOMY_DIR = os.path.join(BASE, '..', 'economy')
MASTER_PATH = os.path.join(BASE, 'master_dataset.csv')

# ── Master dataset (primary source for all modelling) ─────────────────────────
master_df = pd.read_csv(MASTER_PATH)
master_df['year'] = master_df['year'].astype(int)

# ── Individual economy CSVs (for raw / extended series) ───────────────────────

## GDP
gdp_df = pd.read_csv(os.path.join(ECONOMY_DIR, 'GDP.csv'))

## Real GDP at constant prices (FRED)
real_gdp_df = pd.read_csv(os.path.join(ECONOMY_DIR, 'Real GDP at Constant National Prices for Sri Lanka.csv'))
real_gdp_df['year'] = pd.to_datetime(real_gdp_df['observation_date']).dt.year
real_gdp_df = real_gdp_df.groupby('year')['RGDPNALKA666NRUG'].mean().reset_index()
real_gdp_df.columns = ['year', 'real_gdp_const']

## Inflation (FRED annual)
inflation_df = pd.read_csv(os.path.join(ECONOMY_DIR, 'Inflation, consumer prices for Sri Lanka.csv'))
inflation_df['year'] = pd.to_datetime(inflation_df['observation_date']).dt.year
inflation_df = inflation_df[['year', 'FPCPITOTLZGLKA']].rename(columns={'FPCPITOTLZGLKA': 'inflation_pct'})

## CPI
cpi_df = pd.read_csv(os.path.join(ECONOMY_DIR, 'Consumer Price Index for Sri Lanka.csv'))

## Remittances (World Bank wide format → long, received USD)
_rem_raw = pd.read_csv(os.path.join(ECONOMY_DIR, 'remittance.csv'))
_year_cols = [c for c in _rem_raw.columns if c.strip()[:4].isdigit()]
_rem_row = _rem_raw[_rem_raw['Series Name'].str.contains(r'received \(current', na=False, regex=True)].iloc[0]
remittance_df = pd.DataFrame({
    'year': [int(c.split(' ')[0]) for c in _year_cols],
    'remittances_usd': [
        float(_rem_row[c]) if _rem_row[c] not in ['..', ''] else np.nan
        for c in _year_cols
    ]
}).dropna()
remittance_df['year'] = remittance_df['year'].astype(int)

## Agricultural output index (FAO composite — mean across all crops)
_agri_raw = pd.read_csv(os.path.join(ECONOMY_DIR, 'agricultural_output_index.csv'))
agri_df = (
    _agri_raw[_agri_raw['Element'] == 'Gross Production Index Number (2014-2016 = 100)']
    .groupby('Year')['Value'].mean()
    .reset_index()
    .rename(columns={'Year': 'year', 'Value': 'agri_output_index'})
)
agri_df['year'] = agri_df['year'].astype(int)

## Exchange rate — real annual averages (no imputation, CBSL + FRED daily sources)
## Full series now in master_dataset; also available as standalone
_fx_early = pd.read_csv(
    os.path.join(ECONOMY_DIR, 'Sri Lankan Rupees to U.S. Dollar Spot Exchange Rate - 2015_2021Mar14.csv')
)
_fx_late  = pd.read_csv(
    os.path.join(ECONOMY_DIR, 'Sri Lankan Rupees to U.S. Dollar Spot Exchange Rate.csv')
)
for _fx in [_fx_early, _fx_late]:
    _fx['date'] = pd.to_datetime(_fx['observation_date'])
    _fx['year'] = _fx['date'].dt.year
_fx_combined = pd.concat([_fx_early[['year','DEXSLUS']], _fx_late[['year','DEXSLUS']]]).dropna()
exchange_rate_df = (
    _fx_combined.groupby('year')['DEXSLUS'].mean()
    .reset_index()
    .rename(columns={'DEXSLUS': 'exchange_rate_lkr_usd'})
)
exchange_rate_df['year'] = exchange_rate_df['year'].astype(int)

## Internet users / population (auxiliary)
try:
    internet_user_df = pd.read_csv(os.path.join(ECONOMY_DIR, 'Internet users for Sri Lank.csv'))
except FileNotFoundError:
    internet_user_df = pd.DataFrame()

try:
    population_df = pd.read_csv(os.path.join(ECONOMY_DIR, 'Populaton total.csv'))
except FileNotFoundError:
    population_df = pd.DataFrame()


# ── Convenience: study-period slice (2015-2024) ───────────────────────────────
STUDY_YEARS = list(range(2015, 2025))

def get_study_period(df, year_col='year'):
    """Filter any DataFrame to the 2015-2024 study window."""
    return df[df[year_col].isin(STUDY_YEARS)].copy().reset_index(drop=True)


# ── Variable catalogue (for downstream scripts) ───────────────────────────────
MACRO_PREDICTORS = [
    'gdp_growth_pct',
    'inflation_cpi_pct',
    'exchange_rate_lkr_usd',
    'remittances_usd',
    'agri_output_index',
]

LABOUR_PREDICTORS = [
    'informal_emp_pct',
    'parttime_emp_pct',
    'youth_lfpr_pct',
    'discouraged_seekers_n',
]

TARGET_VARS = [
    'underemployment_total',    # hours-based (DCS LFS)
    'tru_female',               # time-related underemployment, female (ILO)
    'tru_male',                 # time-related underemployment, male (ILO)
]

ALL_PREDICTORS = MACRO_PREDICTORS + LABOUR_PREDICTORS


if __name__ == '__main__':
    print("Master dataset:", master_df.shape)
    print("Columns:", list(master_df.columns))
    print("\nStudy period (2015-2024):")
    print(get_study_period(master_df)[['year'] + TARGET_VARS[:1] + MACRO_PREDICTORS].to_string(index=False))
    print("\nRemittances (2015-2024):")
    print(get_study_period(remittance_df).to_string(index=False))
    print("\nAgri output index (2015-2024):")
    print(get_study_period(agri_df).to_string(index=False))
    print("\nExchange rate (2015-2024, real):")
    print(get_study_period(exchange_rate_df).to_string(index=False))
