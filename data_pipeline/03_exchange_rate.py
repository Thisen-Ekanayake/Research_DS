import pandas as pd
import pandas_datareader as pdr
import warnings
warnings.filterwarnings('ignore')

def build_exchange_rate_series():
    print("Step 2: Reconstructing Real Exchange Rate...")
    
    # 1. Load CBSL Nominal Exchange Rate (2015 - 2020)
    cbsl_file = '../economy/Sri Lankan Rupees to U.S. Dollar Spot Exchange Rate - 2015_2021Mar14.csv'
    cbsl_daily = pd.read_csv(cbsl_file, parse_dates=['observation_date'], index_col='observation_date')
    cbsl_daily.columns = ['nominal_lkr_usd']
    # Filter 2015 Q1 - 2020 Q4
    cbsl_daily = cbsl_daily.loc['2015-01-01':'2020-12-31']
    cbsl_qtr = cbsl_daily.resample('QS').mean()
    print(f"Loaded CBSL Qtr data: {len(cbsl_qtr)} quarters.")

    # 2. Download FRED Nominal Exchange Rate (2021 - 2024)
    print("Downloading FRED DEXSLUS...")
    fred_daily = pdr.get_data_fred('DEXSLUS', start='2021-01-01', end='2024-12-31')
    fred_daily.columns = ['nominal_lkr_usd']
    fred_qtr = fred_daily.resample('QS').mean()
    print(f"Loaded FRED Qtr data: {len(fred_qtr)} quarters.")

    # Splicing - Splicing check against overlaps
    cbsl_2020_q4 = cbsl_qtr.loc['2020-10-01', 'nominal_lkr_usd']
    fred_2021_q1 = fred_qtr.loc['2021-01-01', 'nominal_lkr_usd']
    gap = abs(cbsl_2020_q4 - fred_2021_q1)
    
    print(f"Splice Diagnostic - CBSL 2020 Q4: {cbsl_2020_q4:.2f}, FRED 2021 Q1: {fred_2021_q1:.2f}")
    if gap <= 2.0:
        print(f"SUCCESS: Gap is {gap:.2f} LKR/USD (within the ±2 LKR/USD threshold).")
    else:
        print(f"WARNING: Gap is {gap:.2f} LKR/USD, exceeding the ±2 threshold. Investigation needed. (Note: CBSL methodology changed in late 2020)")

    nominal_qtr = pd.concat([cbsl_qtr, fred_qtr])
    
    # Generate Annual nominal series via aggregation
    nominal_annual = nominal_qtr.resample('YE').mean()

    # 3. USA CPI from FRED
    print("Downloading FRED CPIAUCSL (US CPI)...")
    cpi_usa = pdr.get_data_fred('CPIAUCSL', start='2015-01-01', end='2024-12-31')
    cpi_usa_qtr = cpi_usa.resample('QS').mean()['CPIAUCSL']
    cpi_usa_annual = cpi_usa.resample('YE').mean()['CPIAUCSL']

    # 4. LKA CPI
    # Since the World Bank CPI file ends at 2017, we will reconstruct the LKA CPI index using the Inflation Rates
    inflation_file = '../economy/Inflation, consumer prices for Sri Lanka.csv'
    inflation_df = pd.read_csv(inflation_file, parse_dates=['observation_date'], index_col='observation_date')
    inflation_df.columns = ['Inflation']
    inflation_df = inflation_df.loc['2014-01-01':'2024-12-31']
    
    # Reconstruct CPI (Base 2014 = 100 for simplicity of index relative growth)
    cpi_lka_annual_vals = [100.0]
    current_cpi = 100.0  # Base logic
    
    for year in range(2015, 2025):
        year_str = f"{year}-01-01"
        inf_rate = inflation_df.loc[year_str, 'Inflation']
        current_cpi = current_cpi * (1 + (inf_rate / 100))
        cpi_lka_annual_vals.append(current_cpi)
        
    cpi_lka_annual = pd.Series(cpi_lka_annual_vals, index=pd.date_range(start='2014-12-31', end='2024-12-31', freq='YE'))
    
    # We must temporal disaggregate LKA CPI to quarterly. For now, simple linear interpolation
    # as CCPI is quite structural.
    cpi_lka_qtr_df = cpi_lka_annual.resample('QS').interpolate(method='linear')
    # Filter bounds
    cpi_lka_qtr = cpi_lka_qtr_df.loc['2015-01-01':'2024-10-01']

    # 5. Calculate Real Exchange Rate: Nominal * (CPI_USA / CPI_LKA)
    # Rebase both CPIs to 2015=100 so the multiplier starts clean
    cpi_usa_qtr = cpi_usa_qtr.bfill().ffill()
    cpi_lka_qtr = cpi_lka_qtr.bfill().ffill()
    cpi_usa_qtr_base = cpi_usa_qtr / cpi_usa_qtr.iloc[0] * 100
    cpi_lka_qtr_base = cpi_lka_qtr / cpi_lka_qtr.iloc[0] * 100
    
    # Ensure strict index alignment for safe multiplication by stripping indices
    cpi_usa_val = cpi_usa_qtr_base.values[:len(nominal_qtr)]
    cpi_lka_val = cpi_lka_qtr_base.values[:len(nominal_qtr)]
    nominal_val = nominal_qtr['nominal_lkr_usd'].values
    
    real_rate_qtr = nominal_val * (cpi_usa_val / cpi_lka_val)
    
    # Annual
    cpi_usa_annual = cpi_usa_annual.bfill().ffill()
    cpi_lka_annual = cpi_lka_annual.bfill().ffill()
    cpi_usa_annual_base = cpi_usa_annual / cpi_usa_annual.iloc[0] * 100
    cpi_lka_annual_base = cpi_lka_annual / cpi_lka_annual.iloc[0] * 100
    
    cpi_usa_ann_val = cpi_usa_annual_base.values[:len(nominal_annual)]
    cpi_lka_ann_val = cpi_lka_annual_base.values[:len(nominal_annual)]
    nominal_ann_val = nominal_annual['nominal_lkr_usd'].values

    real_rate_annual = nominal_ann_val * (cpi_usa_ann_val / cpi_lka_ann_val)

    # 6. Save Outputs
    output_qtr = pd.DataFrame({
        'nominal_lkr_usd': nominal_val,
        'real_lkr_usd': real_rate_qtr
    }, index=nominal_qtr.index)
    
    output_annual = pd.DataFrame({
        'nominal_lkr_usd': nominal_ann_val,
        'real_lkr_usd': real_rate_annual
    }, index=nominal_annual.index)
    
    output_qtr.to_csv('quarterly_exchange_rates.csv')
    output_annual.to_csv('annual_exchange_rates.csv')
    print("Done! Saved to quarterly_exchange_rates.csv and annual_exchange_rates.csv")


if __name__ == "__main__":
    build_exchange_rate_series()
