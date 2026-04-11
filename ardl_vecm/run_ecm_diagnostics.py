"""
run_ecm_diagnostics.py
======================

EXAMPLE: Running the ECM Robustness Diagnostic Suite

This script integrates your baseline ARDL/ECM estimation with the 
comprehensive diagnostic toolkit to diagnose the λ overshooting issue.

Usage:
------
1. Load your baseline ARDL model estimates (from run_ecm.py or equivalent)
2. Extract the long-run coefficients and intercept
3. Run this script to execute all 6 diagnostic steps

The script will:
  - Re-fit ARDL across a grid of lag combinations (BIC-optimal)
  - Diagnose multicollinearity (VIF) and apply Ridge if needed
  - Re-estimate ECM with outlier dummies and robust regression
  - Estimate DOLS alternative
  - Run stability tests (CUSUM, recursive λ, Chow)
  - Generate summary table and academic paragraph

Output:
-------
Files saved to: ardl_vecm/output/ecm_diagnostics/
  - ecm_robustness_summary.csv        (summary table)
  - robustness_paragraph.txt          (3-sentence academic paragraph)
  - cusum_stability_test.png          (CUSUM & CUSUM-SQ plots)
  - recursive_lambda_estimate.png     (recursive λ over expanding windows)
"""

import os
import sys
import pandas as pd
import numpy as np

# Add path to diagnostic toolkit
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from ecm_robustness_diagnostics import main

# ─────────────────────────────────────────────────────────────────────────────
# LOAD YOUR BASELINE ARDL/ECM RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def extract_baseline_results():
    """
    Extract baseline ARDL long-run coefficients and intercept.
    
    EXTRACTED FROM: run_ecm.py output (April 12, 2026)
    
    PART A — LONG-RUN COEFFICIENTS (from ARDL BIC-selected model):
      Long-run coefficients (θ):
        GDP_Growth_Rate: -0.8250
        Inflation_Rate:   -0.1877
        Youth_LFPR:        0.3074
        Remittances_USD:  -0.0000
      
      Const (α_lr): 30.4880
      Implied λ = Σφ - 1 = 0.0000 - 1 = -1.0000
    
    PART B — ECM RESULTS:
      ECT coefficient (speed-of-adjustment): -0.6628 (p=0.0011)
      95% CI: [-1.0607, -0.2649]
    """
    
    # Baseline long-run coefficients from ARDL (PART A output)
    lr_coefs = {
        "GDP_Growth_Rate": -0.8250,     # θ_GDP from ARDL long-run equation
        "Inflation_Rate": -0.1877,      # θ_Infl
        "Youth_LFPR": 0.3074,           # θ_Youth
        "Remittances_USD": -0.0000,     # θ_Remit
    }
    
    alpha_lr = 30.4880                  # α_lr from ARDL const
    
    # Baseline speed-of-adjustment from ARDL AR lag sum
    baseline_lambda = -1.0000           # λ from Σφ - 1 = 0 - 1
    
    print("Baseline ARDL Model Parameters (BIC-selected)")
    print("─" * 70)
    print(f"Long-run coefficients (θ) from PART A:")
    for var, coef in lr_coefs.items():
        print(f"  {var:20s}: {coef:8.4f}")
    print(f"Long-run intercept (α):            {alpha_lr:8.4f}")
    print(f"Implied λ from ARDL (Σφ - 1):      {baseline_lambda:8.4f}")
    print(f"ECM speed-of-adjustment (PART B):  -0.6628  (p=0.0011) [95% CI: -1.061, -0.265]")
    print("─" * 70)
    print("Note: ARDL λ = -1.000 is at stability boundary (|λ|=1).")
    print("      ECM λ = -0.6628 is robustly stable (|λ|<1).")
    print("      Diagnostics will verify stability across alternative specs.")
    print("─" * 70)
    
    return lr_coefs, alpha_lr, baseline_lambda


def main_run():
    """Execute the diagnostic suite."""
    
    print("\n" + "="*80)
    print("ECM ROBUSTNESS DIAGNOSTICS — Sri Lanka Underemployment ARDL/ECM")
    print("="*80 + "\n")
    
    # Extract baseline results
    lr_coefs, alpha_lr, baseline_lambda = extract_baseline_results()
    
    # Data file
    data_file = os.path.join(BASE, "quarterly_master_dataset.csv")
    
    if not os.path.exists(data_file):
        print(f"ERROR: Data file not found: {data_file}")
        print("Please ensure 'quarterly_master_dataset.csv' is in the ardl_vecm/ directory")
        sys.exit(1)
    
    print(f"\nData file: {data_file}")
    print(f"Executing diagnostic suite...\n")
    
    # Run full diagnostic suite
    main(
        data_filepath=data_file,
        lr_coefs_baseline=lr_coefs,
        alpha_lr_baseline=alpha_lr,
        baseline_lambda=baseline_lambda,
    )
    
    print("\n✓ DIAGNOSTICS COMPLETE")
    print("\nNext steps:")
    print("  1. Review ecm_robustness_summary.csv for specification comparison")
    print("  2. Examine plots: cusum_stability_test.png, recursive_lambda_estimate.png")
    print("  3. Read robustness_paragraph.txt for academic discussion")
    print("  4. Update your paper's robustness section with the generated paragraph")


if __name__ == "__main__":
    main_run()
