import nbformat as nbf
import os
from pathlib import Path

nb = nbf.v4.new_notebook()

# Markdown cells
md_intro = nbf.v4.new_markdown_cell("""# Sensitivity Analysis: Underemployment Definitions

**Project:** Analysing the Economic Drivers of Underemployment in Sri Lanka (2015-2024)

This notebook addresses **Phase 5 / Final Outputs** of the research proposal:
> *A sensitivity analysis comparing findings across both underemployment definitions.*

## Definitions Analysed:
1. **Time-Related Underemployment (`Underemployment_Rate`)**: Employed persons working <40 hours/week who desire more hours. This is the primary target variable used in the core ARDL and XGBoost models.
2. **Qualification-Based Underemployment (`Qual_Underemployment_Rate`)**: Employed persons engaged in occupations with required skill levels below their educational attainment (occupational mismatch).

## Objectives:
1. Merge the two extracted series into a unified dataset.
2. Perform comparative EDA to visualize their independent trajectories.
3. Generate comparative correlation heatmaps against macroeconomic indicators.
4. Fit parallel XGBoost models to compare SHAP feature importance rankings for both definitions.""")

md_setup = nbf.v4.new_markdown_cell("""## 1. Data Setup & Integration""")

code_setup = nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap

# Set styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('deep')

# Define paths
import os
BASE_DIR = '..'
MASTER_DATA_PATH = os.path.join(BASE_DIR, 'ardl_vecm', 'master_dataset.csv')
QUAL_DATA_PATH = os.path.join(BASE_DIR, 'extraction', 'qualification_underemployment.csv')

# Load datasets
master_df = pd.read_csv(MASTER_DATA_PATH)
qual_df = pd.read_csv(QUAL_DATA_PATH)

# The master_dataset has lowercase column names in some phases, let's normalize
master_df.columns = [str(c).title().replace('_Pct', '_pct').replace('_Rate', '_Rate') for c in master_df.columns]
if 'Year' not in master_df.columns and 'Date' in master_df.columns:
    master_df['Year'] = pd.to_datetime(master_df['Date']).dt.year

print("Master Dataset columns:", list(master_df.columns))

# Ensure year is standard integer
master_df['Year'] = master_df['Year'].astype(int)
qual_df['Year'] = qual_df['Year'].astype(int)

# Merge datasets
df = pd.merge(master_df, qual_df[['Year', 'Qual_Underemployment_Rate']], on='Year', how='inner')

# Identify macro predictors
macro_cols = [
    'Gdp_Growth_Rate', 
    'Inflation_Rate', 
    'Youth_Lfpr_15_24', 
    'Exchange_Rate_Lkr_Usd',
    'Informal_Pct'
]
# Ensure macro columns exist, handle naming variations
actual_macro_cols = []
for expected in macro_cols:
    matched = False
    for c in df.columns:
        if expected.lower() == c.lower() or expected.replace('_', '').lower() == c.replace('_', '').lower():
            actual_macro_cols.append(c)
            matched = True
            break
    if not matched:
        print(f"Warning: Macro column roughly matching '{expected}' not found.")

print("Using macro predictors:", actual_macro_cols)

# Define our two target variables
targets = ['Underemployment_Rate', 'Qual_Underemployment_Rate']

df[targets + actual_macro_cols].head()""")

md_eda = nbf.v4.new_markdown_cell("""## 2. Comparative EDA""")

code_eda = nbf.v4.new_code_cell("""# 1. Timeline Comparison
fig, ax1 = plt.subplots(figsize=(12, 6))

color1 = 'tab:blue'
ax1.set_xlabel('Year', fontweight='bold')
ax1.set_ylabel('Time-Related Rate (%)', color=color1, fontweight='bold')
ax1.plot(df['Year'], df['Underemployment_Rate'], marker='o', color=color1, linewidth=2, label='Time-Related (Left)')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 10) # Time-related is usually ~2-8%

# Highlight 2022 Crisis
ax1.axvspan(2021.5, 2022.5, color='gray', alpha=0.2, label='2022 Economic Crisis')

ax2 = ax1.twinx()  
color2 = 'tab:red'
ax2.set_ylabel('Qualification-Based Rate (%)', color=color2, fontweight='bold')  
ax2.plot(df['Year'], df['Qual_Underemployment_Rate'], marker='s', color=color2, linewidth=2, linestyle='--', label='Qualification-Based (Right)')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(60, 75) # Qual-based is usually ~64-71%

plt.title('Divergent Scales: Time-Related vs Qualification-Based Underemployment', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.show()""")

md_corr = nbf.v4.new_markdown_cell("""## 3. Comparative Correlation Matrix

Do macroeconomic shocks correlate similarly with both metrics?""")

code_corr = nbf.v4.new_code_cell("""# Calculate correlation
corr_df = df[actual_macro_cols + targets].corr()

# Extract just the targets vs macro part
target_corr = corr_df.loc[actual_macro_cols, targets]

plt.figure(figsize=(10, 6))
sns.heatmap(target_corr, annot=True, cmap='RdBu', center=0, vmin=-1, vmax=1, fmt='.2f', 
            linewidths=1, annot_kws={'size': 12, 'weight': 'bold'})
plt.title('Macroeconomic Correlation: Time-Related vs Qualification-Based', fontsize=14, pad=20)
plt.ylabel('Macroeconomic Indicator')
plt.xlabel('Underemployment Definition')
plt.tight_layout()
plt.show()""")

md_shap = nbf.v4.new_markdown_cell("""## 4. XGBoost & SHAP Parallel Models

Let's fit the exact same XGBoost architecture to both target definitions and extract the SHAP values to see how feature importance and directionality change.""")

code_shap = nbf.v4.new_code_cell("""# Prepare X and y
X = df[actual_macro_cols].copy()
y_time = df['Underemployment_Rate']
y_qual = df['Qual_Underemployment_Rate']

# Fit XGBoost for Time-Related
model_time = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
model_time.fit(X, y_time)

# Fit XGBoost for Qualification-Based
model_qual = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
model_qual.fit(X, y_qual)

# Calculate SHAP values
explainer_time = shap.TreeExplainer(model_time)
shap_values_time = explainer_time(X)

explainer_qual = shap.TreeExplainer(model_qual)
shap_values_qual = explainer_qual(X)

# Compute mean absolute SHAP for ranking
shap_abs_time = np.abs(shap_values_time.values).mean(axis=0)
shap_abs_qual = np.abs(shap_values_qual.values).mean(axis=0)

# Create a DataFrame for comparison
shap_comparison = pd.DataFrame({
    'Feature': actual_macro_cols,
    'Time_Related_Importance': shap_abs_time,
    'Qual_Based_Importance': shap_abs_qual
})

# Normalize within each model to percentages for fair cross-model comparison
shap_comparison['Time_Rel_Pct'] = (shap_comparison['Time_Related_Importance'] / shap_comparison['Time_Related_Importance'].sum()) * 100
shap_comparison['Qual_Based_Pct'] = (shap_comparison['Qual_Based_Importance'] / shap_comparison['Qual_Based_Importance'].sum()) * 100

shap_comparison = shap_comparison.sort_values('Time_Rel_Pct', ascending=True)

# Plotting side-by-side Horizontal Bar Chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

# Ensure sorting by the index we just created
features = shap_comparison['Feature']
time_pcts = shap_comparison['Time_Rel_Pct']
qual_pcts = shap_comparison['Qual_Based_Pct']

y_pos = np.arange(len(features))

ax1.barh(y_pos, time_pcts, color='tab:blue')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(features)
ax1.set_xlabel('Relative Importance Contribution (%)')
ax1.set_title('Time-Related Drivers (SHAP)', fontweight='bold')

ax2.barh(y_pos, qual_pcts, color='tab:red')
ax2.set_xlabel('Relative Importance Contribution (%)')
ax2.set_title('Qualification-Based Drivers (SHAP)', fontweight='bold')

for ax in [ax1, ax2]:
    ax.set_xlim(0, max(time_pcts.max(), qual_pcts.max()) * 1.1)
    
plt.suptitle('XGBoost SHAP Feature Importance: Definition Sensitivity', fontsize=16, y=1.05)
plt.tight_layout()
plt.show()""")

md_conclusion = nbf.v4.new_markdown_cell("""## 5. Summary of Findings (Sensitivity Analysis)

**(Note: Values will be interpreted upon successful execution)**

1. **Scale Sensitivity**: ...
2. **Correlation Sensitivity**: ...
3. **Causal/Predictive Driver Sensitivity**: ...
""")

# Append cells to notebook
nb['cells'] = [md_intro, md_setup, code_setup, md_eda, code_eda, md_corr, code_corr, md_shap, code_shap, md_conclusion]

# Write out the notebook file
target_file = os.path.join("Data_Analysis", "Sensitivity_Analysis_Underemployment_Definitions.ipynb")
with open(target_file, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Successfully generated {target_file}")
