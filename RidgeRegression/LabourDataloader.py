
import pandas as pd


unemployment_df = pd.read_csv('../labour/csv/from1990/API_SL.UEM.TOTL.ZS_DS2_en_csv_v2_93.csv', skiprows=4)
unemployment_df=unemployment_df.set_index('Country Name')
unemployment_df=unemployment_df.loc['Sri Lanka'].dropna()

labour_force_df = pd.read_csv('../labour/csv/from1990/API_SL.TLF.TOTL.IN_DS2_en_csv_v2_761.csv', skiprows=4)
labour_force_df=labour_force_df.set_index('Country Name')
labour_force_df=labour_force_df.loc['Sri Lanka'].dropna()



