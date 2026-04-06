import pandas as pd
import numpy as np
from statsmodels.tsa.interp.denton import dentonm
import os

def disaggregate_remittances():
    print("--- Disaggregating Remittances ---")
    # 1. Load Monthly Indicator (Worker's Remittances)
    ind_df = pd.read_csv('../economy/workers_remittances.csv')
    
    # Clean the monthly data
    # Drop rows like 'Total' or empty
    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    ind_df = ind_df[ind_df['Month'].isin(month_names)].copy()
    
    # Map months to quarters
    month_to_q = {
        'January':'Q1', 'February':'Q1', 'March':'Q1',
        'April':'Q2', 'May':'Q2', 'June':'Q2',
        'July':'Q3', 'August':'Q3', 'September':'Q3',
        'October':'Q4', 'November':'Q4', 'December':'Q4'
    }
    ind_df['Quarter'] = ind_df['Month'].map(month_to_q)
    
    # Melt to long format: ['Month', 'Quarter', 'Year', 'Value']
    years = [c for c in ind_df.columns if c not in ['Month', 'Quarter', 'Unnamed: 18'] and '20' in str(c)]
    melted = pd.melt(ind_df, id_vars=['Month', 'Quarter'], value_vars=years, var_name='Year', value_name='Remittance_USD')
    melted['Year'] = melted['Year'].astype(float).astype(int)
    
    # Aggregate to quarterly
    q_indicator = melted.groupby(['Year', 'Quarter'])['Remittance_USD'].sum().reset_index()
    q_indicator = q_indicator.sort_values(['Year', 'Quarter']).dropna()
    
    # The indicator series is the quarterly sum
    indicator_series = q_indicator['Remittance_USD'].values
    indicator_years = q_indicator['Year'].unique()
    
    # 2. Load Annual Benchmark (WB Data)
    wb_df = pd.read_csv('../economy/remittance.csv')
    
    # Find the received remittances row
    target_row = wb_df[wb_df['Series Name'].str.contains("Personal remittances, received \\(current", na=False)]
    if target_row.empty:
        raise ValueError("Could not find Personal remittances, received row in WB data.")
        
    # Extract benchmark years
    bench_data = []
    for col in target_row.columns:
        if '[YR' in col:
            yr_str = col.split(' ')[0]
            try:
                yr = int(yr_str)
                val_str = target_row.iloc[0][col]
                if pd.notna(val_str) and val_str != '..':
                    bench_data.append({'Year': yr, 'WB_Remittance': float(val_str)})
            except:
                pass
                
    bench_df = pd.DataFrame(bench_data).sort_values('Year')
    
    # Filter both down to overlapping years
    common_years = sorted(list(set(indicator_years).intersection(set(bench_df['Year'].unique()))))
    print(f"Overlapping years for Remittances: {common_years[0]} to {common_years[-1]}")
    
    bench_overlap = bench_df[bench_df['Year'].isin(common_years)]
    q_overlap = q_indicator[q_indicator['Year'].isin(common_years)]
    
    benchmark_series = bench_overlap['WB_Remittance'].values
    indicator_overlap_series = q_overlap['Remittance_USD'].values
    
    # 3. Denton Disaggregation
    # Note: dentonm requires indicator to line up with the first benchmark, so we matched the years.
    # Since benchmark is Annual and Indicator is Quarterly, we use freq="aq"
    disaggregated = dentonm(indicator_overlap_series, benchmark_series, freq="aq")
    
    q_overlap['Disaggregated_WB_Remittance'] = disaggregated
    
    # Verify sum
    for y in common_years:
        q_sum = q_overlap[q_overlap['Year'] == y]['Disaggregated_WB_Remittance'].sum()
        a_val = bench_overlap[bench_overlap['Year'] == y]['WB_Remittance'].values[0]
        assert np.isclose(q_sum, a_val, rtol=1e-4), f"Mismatch in year {y}: {q_sum} vs {a_val}"
    
    print("Remittance Denton disaggregation verified (Sum(Q) == Annual).")
    return q_overlap


def disaggregate_agriculture():
    print("\n--- Disaggregating Agriculture Output ---")
    
    # 1. Load Semi-Annual Indicators
    maha_df = pd.read_csv('../economy/paddy_extent_maha_season.csv')
    yala_df = pd.read_csv('../economy/paddy_extent_yala_season.csv')
    
    # Clean Maha (Harvested in Q1)
    maha_data = []
    for _, row in maha_df.iterrows():
        try:
            # Year is like '2015/16' -> Harvest is in 2016
            yr_str = str(row['Year'])
            if '/' in yr_str:
                yr = int(yr_str.split('/')[0]) + 1
                if yr < 100: yr += 1900 # fix 99 -> 1999
                if yr < 2000 and int(yr_str.split('/')[0]) >= 2000: yr += 2000 # wait, 1999/00 -> 2000
                if int(yr_str.split('/')[0]) == 1999: yr = 2000
                # safer parse:
                p1 = int(yr_str.split('/')[0])
                yr = p1 + 1
                
                prod = float(str(row['Production (000 Mt)']).replace(',', ''))
                if pd.notna(prod):
                    maha_data.append({'Year': yr, 'Maha_Prod': prod})
        except:
            pass
    maha_clean = pd.DataFrame(maha_data)
    
    # Clean Yala (Harvested in Q3)
    yala_data = []
    for _, row in yala_df.iterrows():
        try:
            yr = int(row['Year'])
            prod = float(str(row['Production (000 Mt)']).replace(',', ''))
            if pd.notna(prod):
                yala_data.append({'Year': yr, 'Yala_Prod': prod})
        except:
            pass
    yala_clean = pd.DataFrame(yala_data)
    
    # 2. Build Artificial Quarterly Indicator
    # Q1 = Maha, Q3 = Yala. Q2, Q4 = small nominal baseline to prevent zeros (say, 5% of average harvest)
    years = sorted(list(set(maha_clean['Year']).intersection(set(yala_clean['Year']))))
    q_data = []
    
    avg_harvest = maha_clean['Maha_Prod'].mean() + yala_clean['Yala_Prod'].mean()
    baseline = avg_harvest * 0.05
    
    for y in years:
        m_val = maha_clean[maha_clean['Year']==y]['Maha_Prod'].values
        m_val = m_val[0] if len(m_val)>0 else baseline
        
        y_val = yala_clean[yala_clean['Year']==y]['Yala_Prod'].values
        y_val = y_val[0] if len(y_val)>0 else baseline
        
        q_data.append({'Year': y, 'Quarter': 'Q1', 'Indicator': m_val})
        q_data.append({'Year': y, 'Quarter': 'Q2', 'Indicator': baseline})
        q_data.append({'Year': y, 'Quarter': 'Q3', 'Indicator': y_val})
        q_data.append({'Year': y, 'Quarter': 'Q4', 'Indicator': baseline})
        
    ind_df = pd.DataFrame(q_data)
    
    # 3. Load Annual Benchmark (FAO Index)
    fao_df = pd.read_csv('../economy/agricultural_output_index.csv')
    fao_agri = fao_df[(fao_df['Item'] == 'Agriculture') & (fao_df['Element'].str.startswith('Gross Production Index'))]
    
    bench_data = []
    for _, row in fao_agri.iterrows():
        yr = int(row['Year'])
        val = float(row['Value'])
        bench_data.append({'Year': yr, 'FAO_Index': val})
    bench_df = pd.DataFrame(bench_data).sort_values('Year')
    
    # Filter to overlapping
    common_years = sorted(list(set(years).intersection(set(bench_df['Year']))))
    print(f"Overlapping years for Agriculture: {common_years[0]} to {common_years[-1]}")
    
    bench_overlap = bench_df[bench_df['Year'].isin(common_years)]
    ind_overlap = ind_df[ind_df['Year'].isin(common_years)].copy()
    
    # Because FAO is an INDEX, Annual = Average(Q1, Q2, Q3, Q4)
    # The Denton method forces Sum(Q) = Annual by default.
    # So we must multiply the Annual benchmark by 4 before denton interpolation.
    benchmark_series = bench_overlap['FAO_Index'].values * 4.0
    indicator_series = ind_overlap['Indicator'].values
    
    disaggregated = dentonm(indicator_series, benchmark_series, freq="aq")
    
    ind_overlap['FAO_Index_Quarterly'] = disaggregated
    
    # Verify index average
    for y in common_years:
        q_avg = ind_overlap[ind_overlap['Year'] == y]['FAO_Index_Quarterly'].mean() # Since we want Average(Q)=Annual
        # Wait, if we interpolated to benchmark * 4, then sum(Q) = Annual*4, so mean(Q) = Annual
        # Let's verify sum then divide by 4.
        q_sum = ind_overlap[ind_overlap['Year'] == y]['FAO_Index_Quarterly'].sum()
        a_val = bench_overlap[bench_overlap['Year'] == y]['FAO_Index'].values[0]
        
        # Verify that the average of the 4 quarters equals the annual index
        assert np.isclose(q_sum / 4.0, a_val, rtol=1e-4), f"Mismatch in year {y}: {q_sum/4.0} vs {a_val}"

    print("Agriculture Denton disaggregation verified (Mean(Q) == Annual).")
    return ind_overlap


if __name__ == "__main__":
    os.makedirs('output', exist_ok=True)
    
    # 1
    remit_q = disaggregate_remittances()
    remit_q.to_csv('output/quarterly_remittances.csv', index=False)
    
    # 2
    agri_q = disaggregate_agriculture()
    agri_q.to_csv('output/quarterly_agricultural_output.csv', index=False)
    
    print("\nSuccessfully exported:")
    print("- data_pipeline/output/quarterly_remittances.csv")
    print("- data_pipeline/output/quarterly_agricultural_output.csv")
