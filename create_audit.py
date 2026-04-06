import pandas as pd

def audit_dataset():
    file_path = 'DataLoader/master_dataset.csv'
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Could not read {file_path}: {e}")
        return
    
    print("# Data Audit Report")
    print(f"Total rows (Years/Quarters): {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Date range: {df['Year'].min()} - {df['Year'].max()}\n")
    
    print("## Variable List & Missing Values")
    missing_data = df.isnull().sum()
    
    for col in df.columns[:20]:  # Just do first 20 for core macro metrics to avoid clutter
        missing = missing_data[col]
        print(f"- **{col}**: {missing} missing values")
        
    print("\n## Imputations Needed (Missing = True)")
    for col in df.columns:
        if missing_data[col] > 0:
            print(f"- {col}: Missing {missing_data[col]} points")

if __name__ == '__main__':
    audit_dataset()
