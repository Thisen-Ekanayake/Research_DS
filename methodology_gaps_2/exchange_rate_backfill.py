"""
exchange_rate_backfill.py
=========================
Replaces KNN-imputed exchange_rate_lkr_usd values for 2015-2020 in
master_dataset.csv with real annual averages computed from the CBSL
daily series (Sri_Lankan_Rupees_to_U_S__Dollar_Spot_Exchange_Rate_-_2015_2021Mar14.csv)
combined with the FRED daily series (Sri_Lankan_Rupees_to_U_S__Dollar_Spot_Exchange_Rate.csv).

Output: master_dataset.csv with exchange_rate_imputed flag = 0 for all years.
"""

import pandas as pd
import numpy as np

MASTER   = '/mnt/user-data/outputs/master_dataset.csv'
EARLY_FX = '/mnt/user-data/uploads/Sri_Lankan_Rupees_to_U_S__Dollar_Spot_Exchange_Rate_-_2015_2021Mar14.csv'
LATE_FX  = '/mnt/user-data/uploads/Sri_Lankan_Rupees_to_U_S__Dollar_Spot_Exchange_Rate.csv'
OUT      = '/mnt/user-data/outputs/master_dataset.csv'

# ── 1. Build complete daily FX series 2015–2024 ──────────────────────────────
early = pd.read_csv(EARLY_FX)
early['date'] = pd.to_datetime(early['observation_date'])
early = early.rename(columns={'DEXSLUS': 'fx'})[['date','fx']].dropna()

late = pd.read_csv(LATE_FX)
late['date'] = pd.to_datetime(late['observation_date'])
late = late.rename(columns={'DEXSLUS': 'fx'})[['date','fx']].dropna()

# Combine; keep early where overlap exists (CBSL is authoritative for 2015-2020)
combined = pd.concat([early, late]).drop_duplicates(subset='date', keep='first')
combined['year'] = combined['date'].dt.year
annual_fx = combined.groupby('year')['fx'].mean().reset_index()
annual_fx.columns = ['year', 'exchange_rate_lkr_usd']

print("Annual exchange rate averages (2015-2024):")
print(annual_fx[annual_fx['year'].between(2015, 2024)].to_string(index=False))

# ── 2. Merge into master dataset ─────────────────────────────────────────────
master = pd.read_csv(MASTER)
master['year'] = master['year'].astype(int)

# Drop old imputed column and exchange rate
master = master.drop(columns=['exchange_rate_lkr_usd', 'exchange_rate_imputed'], errors='ignore')

# Merge real values
master = master.merge(
    annual_fx[annual_fx['year'].between(2015, 2024)],
    on='year', how='left'
)
master['exchange_rate_imputed'] = 0   # all real now

master = master.sort_values('year').reset_index(drop=True)
master.to_csv(OUT, index=False)

print(f"\nBackfill complete. exchange_rate_imputed = 0 for all {len(master)} rows.")
print(f"Saved → {OUT}")
print(master[['year','exchange_rate_lkr_usd','exchange_rate_imputed']].to_string(index=False))
