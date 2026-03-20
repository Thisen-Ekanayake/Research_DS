#!/usr/bin/env python
"""
Lagged Effects Analysis: Testing Lag Periods for Economic Features
Tests if previous years' economic data predicts current unemployment better
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Setup
base_path = Path('/Users/janudax/Computer_Science/Uom stuff/Reaserch_DS')
labour_path = base_path / 'labour' / 'csv'
economy_path = base_path / 'economy'

print("=" * 80)
print("LAGGED EFFECTS ANALYSIS: Economic Features vs Unemployment")
print("=" * 80)

# ============================================================
# LOAD DATA
# ============================================================
print("\n[1/6] Loading data...")

# Load unemployment
unemp_file = labour_path / 'API_SL.UEM.TOTL.ZS_DS2_en_csv_v2_93.csv'
unemp_df = pd.read_csv(unemp_file, skiprows=4)
year_cols = [str(y) for y in range(1990, 2024)]
unemp_long = unemp_df[['Country Name', 'Indicator Name'] + year_cols].melt(
    id_vars=['Country Name', 'Indicator Name'],
    var_name='Year',
    value_name='Unemployment_Rate'
)
unemp_long['Year'] = pd.to_numeric(unemp_long['Year'], errors='coerce')
unemp_long = unemp_long.dropna(subset=['Unemployment_Rate'])

# Load economic data
economic_files = {
    'GDP': 'Gross Domestic Product for Sri Lanka.csv',
    'Inflation': 'Inflation, consumer prices for Sri Lanka.csv',
    'CPI': 'Consumer Price Index for Sri Lanka.csv',
    'GNI': 'Gross National Income for Sri Lanka.csv',
    'Real_GDP': 'Real GDP at Constant National Prices for Sri Lanka.csv',
}

economic_data = {}
for name, fn in economic_files.items():
    try:
        df = pd.read_csv(economy_path / fn)
        if 'observation_date' in df.columns:
            df['Year'] = pd.to_datetime(df['observation_date']).dt.year
            vc = df.columns[1]
            ed = df[['Year', vc]].dropna()
            ey = ed.groupby('Year')[vc].mean().reset_index()
            ey.columns = ['Year', name]
            economic_data[name] = ey
    except:
        pass

# Merge
df_merged = unemp_long[['Year', 'Unemployment_Rate']].copy().dropna()
for name, ed in economic_data.items():
    df_merged = df_merged.merge(ed, on='Year', how='inner')

df_merged = df_merged.sort_values('Year').reset_index(drop=True)
feat_cols = [c for c in df_merged.columns if c not in ['Year', 'Unemployment_Rate']]

print(f"✓ Data loaded: Years {df_merged['Year'].min():.0f}-{df_merged['Year'].max():.0f}")
print(f"✓ Features: {feat_cols}")
print(f"✓ Records: {len(df_merged)}")

# ============================================================
# CREATE LAGGED FEATURES
# ============================================================
print("\n[2/6] Creating lagged features...")

# Test different lag periods
lag_periods = [0, 1, 2, 3]  # 0 = no lag (current year), 1 = 1-year lag, etc.

results_by_lag = {}

for lag in lag_periods:
    print(f"\n  Testing LAG={lag} year(s)...")
    
    # Create lagged features
    df_lagged = df_merged.copy()
    
    if lag > 0:
        # Shift economic features back by lag years
        # (Year N's unemployment explained by Year N-lag economic data)
        for col in feat_cols:
            df_lagged[f'{col}_lag{lag}'] = df_lagged[col].shift(lag)
        
        # Remove rows with NaN from lagging
        df_lagged = df_lagged.dropna()
        
        # Use lagged features
        feature_cols_to_use = [f'{col}_lag{lag}' for col in feat_cols]
    else:
        # Current year features (no lag)
        feature_cols_to_use = feat_cols.copy()
        df_lagged = df_lagged.dropna()
    
    if len(df_lagged) < 10:
        print(f"    ⚠ Insufficient data after lagging: {len(df_lagged)} records")
        continue
    
    X = df_lagged[feature_cols_to_use].copy()
    y = df_lagged['Unemployment_Rate'].copy()
    
    # Ensure numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(X.mean())
    X = X.astype(float)
    y = y.astype(float)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols_to_use)
    
    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train models
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    gb_model.fit(X_tr, y_tr)
    
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_tr, y_tr)
    
    # Evaluate
    y_pred_gb = gb_model.predict(X_te)
    y_pred_rf = rf_model.predict(X_te)
    
    r2_gb = r2_score(y_te, y_pred_gb)
    r2_rf = r2_score(y_te, y_pred_rf)
    rmse_gb = np.sqrt(mean_squared_error(y_te, y_pred_gb))
    rmse_rf = np.sqrt(mean_squared_error(y_te, y_pred_rf))
    mae_gb = mean_absolute_error(y_te, y_pred_gb)
    mae_rf = mean_absolute_error(y_te, y_pred_rf)
    
    results_by_lag[lag] = {
        'gb_r2': r2_gb,
        'rf_r2': r2_rf,
        'gb_rmse': rmse_gb,
        'rf_rmse': rmse_rf,
        'gb_mae': mae_gb,
        'rf_mae': mae_rf,
        'gb_model': gb_model,
        'rf_model': rf_model,
        'n_samples': len(X_te),
        'X_scaled': X_scaled,
        'y': y,
        'feature_names': feature_cols_to_use
    }
    
    print(f"    GB: R²={r2_gb:.4f}, RMSE={rmse_gb:.4f}")
    print(f"    RF: R²={r2_rf:.4f}, RMSE={rmse_rf:.4f}")

# ============================================================
# COMPARE RESULTS
# ============================================================
print("\n" + "=" * 80)
print("RESULTS SUMMARY: Model Performance by Lag Period")
print("=" * 80)

results_df = pd.DataFrame([
    {
        'Lag (years)': lag,
        'GB_R2': results_by_lag[lag]['gb_r2'],
        'RF_R2': results_by_lag[lag]['rf_r2'],
        'GB_RMSE': results_by_lag[lag]['gb_rmse'],
        'RF_RMSE': results_by_lag[lag]['rf_rmse'],
        'GB_MAE': results_by_lag[lag]['gb_mae'],
        'RF_MAE': results_by_lag[lag]['rf_mae'],
        'Samples': results_by_lag[lag]['n_samples']
    }
    for lag in sorted(results_by_lag.keys())
])

print("\n" + results_df.to_string(index=False))

# Find best lag
best_lag_gb = results_df.loc[results_df['GB_R2'].idxmax(), 'Lag (years)']
best_lag_rf = results_df.loc[results_df['RF_R2'].idxmax(), 'Lag (years)']
best_r2_gb = results_df['GB_R2'].max()
best_r2_rf = results_df['RF_R2'].max()

print(f"\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print(f"\n✓ Best Gradient Boosting: LAG={int(best_lag_gb)} years with R²={best_r2_gb:.4f}")
print(f"✓ Best Random Forest: LAG={int(best_lag_rf)} years with R²={best_r2_rf:.4f}")

improvement_gb = best_r2_gb - results_by_lag[0]['gb_r2']
improvement_rf = best_r2_rf - results_by_lag[0]['rf_r2']

print(f"\nImprovement over current year (lag=0):")
print(f"  Gradient Boosting: {improvement_gb:+.4f} ({improvement_gb/abs(results_by_lag[0]['gb_r2'])*100:+.1f}%)")
print(f"  Random Forest: {improvement_rf:+.4f} ({improvement_rf/abs(results_by_lag[0]['rf_r2'])*100:+.1f}%)")

# ============================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================
print(f"\n" + "=" * 80)
print("FEATURE IMPORTANCE: Best Models")
print("=" * 80)

for model_type in ['gb_model', 'rf_model']:
    best_lag_for_model = best_lag_gb if model_type == 'gb_model' else best_lag_rf
    model = results_by_lag[best_lag_for_model][model_type]
    feature_names = results_by_lag[best_lag_for_model]['feature_names']
    
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    model_name = "Gradient Boosting" if model_type == 'gb_model' else "Random Forest"
    print(f"\n{model_name} (Lag={int(best_lag_for_model)}):")
    print(importance_df.to_string(index=False))

# ============================================================
# VISUALIZATIONS
# ============================================================
print(f"\n[3/6] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: R² by Lag
ax = axes[0, 0]
ax.plot(results_df['Lag (years)'], results_df['GB_R2'], 'o-', linewidth=2, markersize=8, label='Gradient Boosting')
ax.plot(results_df['Lag (years)'], results_df['RF_R2'], 's-', linewidth=2, markersize=8, label='Random Forest')
ax.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Lag Period (years)', fontweight='bold')
ax.set_ylabel('R² Score', fontweight='bold')
ax.set_title('Model Performance vs Lag Period', fontweight='bold', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: RMSE by Lag
ax = axes[0, 1]
ax.plot(results_df['Lag (years)'], results_df['GB_RMSE'], 'o-', linewidth=2, markersize=8, label='Gradient Boosting')
ax.plot(results_df['Lag (years)'], results_df['RF_RMSE'], 's-', linewidth=2, markersize=8, label='Random Forest')
ax.set_xlabel('Lag Period (years)', fontweight='bold')
ax.set_ylabel('RMSE (%)', fontweight='bold')
ax.set_title('Prediction Error vs Lag Period', fontweight='bold', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: MAE by Lag
ax = axes[1, 0]
ax.plot(results_df['Lag (years)'], results_df['GB_MAE'], 'o-', linewidth=2, markersize=8, label='Gradient Boosting')
ax.plot(results_df['Lag (years)'], results_df['RF_MAE'], 's-', linewidth=2, markersize=8, label='Random Forest')
ax.set_xlabel('Lag Period (years)', fontweight='bold')
ax.set_ylabel('MAE (%)', fontweight='bold')
ax.set_title('Mean Absolute Error vs Lag Period', fontweight='bold', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Sample Size by Lag
ax = axes[1, 1]
ax.bar(results_df['Lag (years)'], results_df['Samples'], color='steelblue', alpha=0.7)
ax.set_xlabel('Lag Period (years)', fontweight='bold')
ax.set_ylabel('Test Samples', fontweight='bold')
ax.set_title('Data Availability by Lag Period', fontweight='bold', fontsize=12)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(base_path / 'lagged_effects_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: lagged_effects_comparison.png")
plt.close()

# ============================================================
# FEATURE IMPORTANCE VISUALIZATION
# ============================================================
print(f"[4/6] Creating feature importance plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, model_type in enumerate(['gb_model', 'rf_model']):
    best_lag = best_lag_gb if model_type == 'gb_model' else best_lag_rf
    model = results_by_lag[best_lag][model_type]
    feature_names = results_by_lag[best_lag]['feature_names']
    
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    ax = axes[idx]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(importance_df)))
    ax.barh(range(len(importance_df)), importance_df['Importance'], color=colors)
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['Feature'])
    ax.set_xlabel('Feature Importance', fontweight='bold')
    
    model_name = "Gradient Boosting" if model_type == 'gb_model' else "Random Forest"
    ax.set_title(f'{model_name} (Lag={int(best_lag)} years)', fontweight='bold', fontsize=12)
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(base_path / 'lagged_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: lagged_feature_importance.png")
plt.close()

# ============================================================
# DETAILED LAG EFFECTS TABLE
# ============================================================
print(f"\n[5/6] Generating detailed analysis...")

lag_analysis_data = []
for lag in sorted(results_by_lag.keys()):
    lag_analysis_data.append({
        'Lag': f'{lag} years',
        'GB_R2': f"{results_by_lag[lag]['gb_r2']:.4f}",
        'RF_R2': f"{results_by_lag[lag]['rf_r2']:.4f}",
        'GB_RMSE': f"{results_by_lag[lag]['gb_rmse']:.4f}",
        'RF_RMSE': f"{results_by_lag[lag]['rf_rmse']:.4f}",
        'Samples': results_by_lag[lag]['n_samples']
    })

lag_analysis_df = pd.DataFrame(lag_analysis_data)

print("\n" + "=" * 80)
print("DETAILED LAG ANALYSIS")
print("=" * 80)
print("\n" + lag_analysis_df.to_string(index=False))

# ============================================================
# CONCLUSION
# ============================================================
print(f"\n[6/6] Analysis complete!")
print("\n" + "=" * 80)
print("CONCLUSION: Do Lagged Effects Matter?")
print("=" * 80)

if best_r2_gb > 0:
    print(f"\n✓ YES - Lagged models show POSITIVE R² score!")
    print(f"  Using {int(best_lag_gb)}-year lag: R² = {best_r2_gb:.4f}")
    print(f"  This means lagged economic data explains {best_r2_gb*100:.2f}% of unemployment variation")
else:
    print(f"\n✗ NO - Even with lags, R² remains NEGATIVE")
    print(f"  Best result: R² = {best_r2_gb:.4f} at {int(best_lag_gb)}-year lag")
    print(f"  Interpretation: Economic features still insufficient as sole predictor")

if abs(improvement_gb) > 0.05:
    print(f"\n💡 IMPROVEMENT: Lagging helps significantly!")
    print(f"  {improvement_gb:+.4f} R² improvement vs current year")
else:
    print(f"\n⚠ LIMITED IMPROVEMENT: {improvement_gb:+.4f} R² gain from lagging")
    print(f"  Suggests other factors matter more than lag timing")

print(f"\n📊 Data constraint: Only {len(df_merged)} overlapping years available")
print(f"   Lagging reduces usable data; consider time series methods instead")

print("\n" + "=" * 80)
print("✓ LAGGED EFFECTS ANALYSIS COMPLETE")
print("=" * 80)
print("\nOutput files generated:")
print("  - lagged_effects_comparison.png")
print("  - lagged_feature_importance.png")
