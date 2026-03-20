# Master Dataset - Data Description

## Overview
This dataset contains comprehensive economic and labor market indicators for **Sri Lanka** spanning from **2015 to 2024** (10 years). It integrates multiple data sources to analyze relationships between employment dynamics, agricultural production, economic conditions, and remittance flows.

## Dataset Dimensions
- **Observations:** 10 (Annual data from 2015-2024)
- **Variables:** 142 indicators

## Key Variable Categories

### 1. Labor Market Indicators
- **Underemployment:** Overall underemployment rate, by gender (Male/Female)
- **Unemployment:** Overall unemployment rate, by gender, youth unemployment (15-24 years)
- **Labor Force Participation:** Youth LFPR (15-24 age group)

### 2. Macroeconomic Indicators
- **Real GDP:** Real Gross Domestic Product
- **GDP Growth Rate:** Year-on-year growth percentage
- **Inflation Rate:** Consumer price inflation
- **Exchange Rate:** LKR/USD spot rate

### 3. Informal Employment
- **Informal Sector Share:** Overall percentage and by gender (Male/Female)

### 4. Agricultural Production Indices (79 variables)
Production indices for major crops and livestock in Sri Lanka, including:
- **Crops:** Rice, Tea, Coconuts, Cinnamon, Cocoa, Sugar cane, etc.
- **Livestock:** Cattle, Buffalo, Chicken, Goat, Pig, Eggs, Milk
- **Other agricultural products:** Natural rubber, Tobacco, etc.

### 5. International Remittances
- **Personal Remittances Paid:** Current USD
- **Personal Remittances Received:** Current USD and % of GDP
- **Personal Transfers Receipts:** Balance of Payments current USD

## Data Quality Notes
- **2024 Data:** Partially incomplete (missing Real_GDP and GDP_Growth_Rate)
- **Inflation Rate 2024:** -0.4294% (deflation)
- **Exchange Rate Volatility:** Significant depreciation from 135.86 LKR/USD (2015) to 301.68 LKR/USD (2024), particularly sharp in 2022 (321.53)

## Notable Time Periods
- **2015-2019:** Relatively stable economic conditions
- **2020:** COVID-19 impact (GDP decline of -4.62%)
- **2022:** Economic crisis year (GDP decline -7.35%, inflation surge to 49.72%, currency depreciation)
- **2023-2024:** Recovery phase with high inflation moderating

## Potential Use Cases
- **ARDL/VECM Analysis:** Examining cointegration between employment and economic variables
- **Time Series Forecasting:** Predicting unemployment and underemployment trends
- **Agricultural Impact Analysis:** Understanding how production affects employment
- **Remittance Impact Study:** Analyzing relationship between remittances and economic indicators
- **Economic Crisis Analysis:** Studying the 2022 crisis impact on labor markets

## Data Source Context
The dataset appears to be compiled from:
- Sri Lankan Ministry of Labour/Statistics
- World Bank indicators
- Agricultural statistics databases
- Balance of Payments data
