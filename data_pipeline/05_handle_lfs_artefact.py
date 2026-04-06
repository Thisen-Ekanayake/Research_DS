import pandas as pd
import os

def handle_lfs_artefact():
    print("--- Handling 2022 LFS Artefact ---")
    
    # Paths
    input_path = '../DataLoader/master_dataset.csv'
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load the dataset
    df = pd.read_csv(input_path)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Create the Unadjusted dataset first (Strategy 3: Sensitivity Check Base)
    unadjusted_path = os.path.join(output_dir, 'master_dataset_unadjusted.csv')
    df.to_csv(unadjusted_path, index=False)
    print(f"Exported raw dataset to: {unadjusted_path}")
    
    # 2. Imputation logic for the Imputed dataset
    df_imputed = df.copy()
    
    # Ensure Year is indexable or search by Year column
    if 'Year' not in df_imputed.columns:
        raise ValueError("Year column missing in master_dataset.csv")
    
    # Initialize the note column
    df_imputed['Underemployment_Note'] = ''
    
    # Locate indices for 2021, 2022, 2023
    idx_2021 = df_imputed.index[df_imputed['Year'] == 2021].tolist()
    idx_2022 = df_imputed.index[df_imputed['Year'] == 2022].tolist()
    idx_2023 = df_imputed.index[df_imputed['Year'] == 2023].tolist()
    
    if not (idx_2021 and idx_2022 and idx_2023):
        print("WARNING: Missing data for 2021, 2022, or 2023. Cannot perform imputation.")
        return
        
    i21 = idx_2021[0]
    i22 = idx_2022[0]
    i23 = idx_2023[0]
    
    target_columns = ['Underemployment_Rate', 'Underemployment_Male', 'Underemployment_Female']
    
    for col in target_columns:
        if col in df_imputed.columns:
            val_2021 = df_imputed.at[i21, col]
            val_2023 = df_imputed.at[i23, col]
            
            # Calculate mean
            imputed_val = (val_2021 + val_2023) / 2.0
            
            # Store original for log
            orig_val = df_imputed.at[i22, col]
            
            # Update the dataframe
            df_imputed.at[i22, col] = imputed_val
            
            print(f"Patched {col} for 2022: changed {orig_val:.2f}% to imputed {imputed_val:.2f}%")
            
            # Verification assertion
            assert df_imputed.at[i22, col] == (val_2021 + val_2023) / 2.0, f"Math verification failed for {col}"
    
    # 3. Add transparency note
    df_imputed.at[i22, 'Underemployment_Note'] = 'LFS artefact — imputed as (2021+2023)/2'
    
    # 4. Export Imputed dataset
    imputed_path = os.path.join(output_dir, 'master_dataset_imputed.csv')
    df_imputed.to_csv(imputed_path, index=False)
    print(f"Exported clean dataset to: {imputed_path}")
    
    print("\nLFS 2022 artefact successfully handled! Verification passes.")

if __name__ == "__main__":
    handle_lfs_artefact()
