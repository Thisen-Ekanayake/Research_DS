"""
ECM_ROBUSTNESS_SETUP_GUIDE.md
==============================

Comprehensive Setup and Usage Guide for the ARDL/ECM Robustness Diagnostic Toolkit

Contents:
---------
1. Overview of the toolkit
2. Prerequisites and installation
3. How to customize the diagnostics
4. Running the toolkit
5. Interpreting the results
6. Troubleshooting

═══════════════════════════════════════════════════════════════════════════════
1. OVERVIEW OF THE TOOLKIT
═══════════════════════════════════════════════════════════════════════════════

The ECM Robustness Diagnostic Toolkit provides six integrated steps to diagnose
and resolve |λ| > 1 (overshooting) issues in ARDL/ECM models:

  Step 1: LAG ORDER RE-SELECTION
    - Grid search over ARDL(p, q1, q2, q3, q4) combinations
    - Report AIC, BIC, HQ for each specification
    - Extract λ from ARDL AR-lag sum = Σφ - 1
    - Identify BIC-optimal vs. AIC-optimal models
    - Flag which lag orders achieve |λ| < 1 (stable)

  Step 2: MULTICOLLINEARITY DIAGNOSIS
    - Compute VIF for all regressors in long-run level equation
    - Flag VIF > 10 as severe multicollinearity
    - If severe VIF detected, apply Ridge regularization
    - Re-estimate long-run coefficients via Ridge regression
    - Use Ridge coefficients for ECM reconstruction

  Step 3A: OUTLIER-ROBUST RE-ESTIMATION (Additive Outlier Dummies)
    - Add impulse indicator dummies for crisis quarters (2020-Q2, 2021-Q2, 2022-Q2)
    - Re-fit ECM with OLS + HC3 robust standard errors
    - Report ECT coefficient λ with HC3 SE and 95% CI
    - Compare with baseline λ to assess outlier influence

  Step 3B: OUTLIER-ROBUST RE-ESTIMATION (Huber Robust Regression)
    - Use M-estimator (Huber norm) instead of OLS
    - Downweights outliers rather than adding dummies
    - Report ECT coefficient λ with robust SE and 95% CI
    - Compare OLS vs. Huber to quantify outlier impact

  Step 4: DYNAMIC OLS (DOLS) ALTERNATIVE ESTIMATOR
    - Estimate long-run relationship with leads and lags of Δx
    - Extract DOLS long-run coefficients (reduces simultaneity bias)
    - Construct ECT from DOLS residuals
    - Re-estimate ECM with DOLS-based ECT
    - Report DOLS λ as robustness check

  Step 5: STABILITY DIAGNOSTICS
    - CUSUM and CUSUM-squared tests for structural breaks
    - Recursive λ estimation over expanding windows (detect regime changes)
    - Chow breakpoint test at 2022-Q1 (mark of major crisis)
    - Visual identification of when |λ| becomes < 1 (if ever)

  Step 6: SUMMARY TABLE & ACADEMIC PARAGRAPH
    - Comparison table of λ across all 6+ specifications
    - 3-sentence paragraph for paper's robustness section
    - Clear recommendation on preferred specification

═══════════════════════════════════════════════════════════════════════════════
2. PREREQUISITES AND INSTALLATION
═══════════════════════════════════════════════════════════════════════════════

Required Python Packages:
  - pandas >= 1.3
  - numpy >= 1.20
  - scipy >= 1.7
  - statsmodels >= 0.13
  - matplotlib >= 3.3
  - scikit-learn >= 0.24 (for Ridge regression in Step 2)

Installation:
  In your .venv, run:
    pip install pandas numpy scipy statsmodels matplotlib scikit-learn

Verify installation:
  python -c "import statsmodels; print(statsmodels.__version__)"

Data Requirements:
  Your quarterly_master_dataset.csv must contain:
    
    Required columns (exact names):
      - Underemployment_Rate
      - GDP_Growth_Rate
      - Inflation_Rate
      - Youth_LFPR
      - Remittances_USD
    
    Optional (will be created if missing):
      - Seasonal dummy columns (Q1_c, Q2_c, Q3_c)
      - Outlier indicators (Outlier_2020Q2, Outlier_2021Q2, Outlier_2022Q2)

═══════════════════════════════════════════════════════════════════════════════
3. HOW TO CUSTOMIZE THE DIAGNOSTICS
═══════════════════════════════════════════════════════════════════════════════

File: run_ecm_diagnostics.py
Location: ardl_vecm/run_ecm_diagnostics.py

Customization Step 1: Extract Your Baseline Results
  ──────────────────────────────────────────────────
  
  Open run_ecm.py and run it to completion to get:
    - Long-run coefficients (θ) for each regressor
    - Long-run intercept (α_lr)
    - Speed-of-adjustment (λ)
  
  These will be printed as part of "PART A — LONG-RUN COEFFICIENTS"

  Example output from run_ecm.py:
  
    λ (from ARDL lag sum) = -1.087  p=0.003
    
    Long-run coefficients (θ):
      GDP_Growth_Rate          θ = -0.1523
      Inflation_Rate           θ = -0.0832
      Youth_LFPR               θ = -0.1204
      Remittances_USD          θ = 0.0008

Customization Step 2: Update run_ecm_diagnostics.py
  ────────────────────────────────────────────────
  
  In the function extract_baseline_results(), replace:
  
    lr_coefs = {
        "GDP_Growth_Rate": -0.1523,     # ← Replace with YOUR value
        "Inflation_Rate": -0.0832,      # ← Replace with YOUR value
        "Youth_LFPR": -0.1204,          # ← Replace with YOUR value
        "Remittances_USD": 0.0008,      # ← Replace with YOUR value
    }
    
    alpha_lr = 3.4692                   # ← Replace with YOUR α_lr
    baseline_lambda = -1.087            # ← Replace with YOUR λ

Customization Step 3: Configure Crisis Quarters (Optional)
  ──────────────────────────────────────────────────────
  
  In ecm_robustness_diagnostics.py, edit the OUTLIERS tuple:
  
    OUTLIERS = [("2020-Q2", "2020 COVID crisis - Q2 peak impact"),
                ("2021-Q2", "2021 sovereign debt default period"),
                ("2022-Q2", "2022 economic crisis - multiple shocks")]

Customization Step 4: Adjust Grid Search Parameters (Optional)
  ────────────────────────────────────────────────────────
  
  In lag_order_grid_search(...), modify:
  
    for p in range(1, p_max + 1):               # Max lags for dependent var
        for q1, q2, q3, q4 in ...:              # Max lags for each regressor
  
  Default: p ∈ {1, 2, 3}, q ∈ {0, 1, 2}
  Adjust to expand/contract search space (larger = slower computation)

Customization Step 5: Configure VIF Threshold (Optional)
  ────────────────────────────────────────────────────
  
  In multicollinearity_diagnosis(...), edit threshold_vif:
  
    def multicollinearity_diagnosis(df, lr_coefs, alpha_lr, threshold_vif=10):
  
  Default threshold: 10 (standard in econometrics)
  Conservative: 5–8 (flag more variables)
  Liberal: 15–20 (only flag extreme cases)

═══════════════════════════════════════════════════════════════════════════════
4. RUNNING THE TOOLKIT
═══════════════════════════════════════════════════════════════════════════════

Step-by-Step Execution:
  ──────────────────

  Step A: Prepare Your Baseline Results
    $ cd C:\\Users\\ASUS\\Desktop\\Research_DS\\Reaserch_DS\\ardl_vecm
    $ python run_ecm.py
    
    This will output your:
      - ARDL lag order (BIC-selected)
      - Long-run coefficients (θ)
      - Long-run intercept (α)
      - Speed-of-adjustment (λ)
      - ECM results table

  Step B: Record the Baseline Values
    Copy λ, all θ coefficients, and α from the output
    Paste into extract_baseline_results() in run_ecm_diagnostics.py

  Step C: Run the Diagnostic Toolkit
    $ python run_ecm_diagnostics.py
    
    This will consume ~2–5 minutes depending on lag search space.
    Progress will be printed to console:
    
      ================================================================================
      ECM ROBUSTNESS DIAGNOSTICS TOOLKIT
      ================================================================================
      Output directory: C:\Users\ASUS\Desktop\Research_DS\Reaserch_DS\ardl_vecm\output\ecm_diagnostics
      
      STEP 1 — LAG ORDER RE-SELECTION (Grid Search)
      ... [grid search progress] ...
      
      STEP 2 — MULTICOLLINEARITY DIAGNOSIS (VIF Analysis)
      ... [VIF results] ...
      
      [continues for all 6 steps]

  Step D: View Results
    All outputs saved to: ardl_vecm/output/ecm_diagnostics/

Running from Jupyter Notebook (Alternative):
  ──────────────────────────────────────────
  
  In a code cell:
    
    import sys
    sys.path.insert(0, r"C:\Users\ASUS\Desktop\Research_DS\Reaserch_DS\ardl_vecm")
    from ecm_robustness_diagnostics import main
    
    lr_coefs = {"GDP_Growth_Rate": -0.1523, ...}
    alpha_lr = 3.4692
    baseline_lambda = -1.087
    
    data_file = r"C:\Users\ASUS\Desktop\Research_DS\Reaserch_DS\ardl_vecm\quarterly_master_dataset.csv"
    
    main(data_file, lr_coefs, alpha_lr, baseline_lambda)

═══════════════════════════════════════════════════════════════════════════════
5. INTERPRETING THE RESULTS
═══════════════════════════════════════════════════════════════════════════════

Step 1 Output Example: BIC-Optimal Lag Order
  ─────────────────────────────────────────

  Top 10 Models by BIC:
    ARDL_order              AIC        BIC        HQC      lambda  stable
    ARDL(1,0,1,1,0)     123.45   145.23    132.10    -0.456     Yes
    ARDL(1,1,1,1,0)     124.12   148.56    134.20    -0.523     Yes
    ARDL(2,1,1,1,0)     125.01   152.34    136.45    -0.587     Yes
    ARDL(3,1,1,2,0)     126.34   157.89    139.56    -1.087     No  ← baseline
  
  Interpretation:
    - ARDL(1,0,1,1,0) has lowest BIC (145.23) → most parsimonious
    - λ = -0.456, |λ| < 1 ✓ (stable)
    - This specification is preferred on information criterion + stability grounds
    - AIC may prefer higher lag order (less parsimonious)

Step 2 Output Example: Multicollinearity Diagnosis
  ─────────────────────────────────────────────────

  Variance Inflation Factors (Long-Run Level Equation):
    Variable          VIF
    Remittances_USD   12.34  ← HIGH (VIF > 10)
    Youth_LFPR         8.12
    Inflation_Rate     5.43
    GDP_Growth_Rate    3.21

  Interpretation:
    - Remittances_USD shows severe multicollinearity (VIF = 12.34)
    - Ridge regression recommended
    - Optimal Ridge penalty λ = 0.0234
    - Ridge coefficients will be used for ECT reconstruction

Step 3A Output Example: Outlier Dummies
  ──────────────────────────────────────

  ECM with Outlier Dummies (OLS, HC3 SE):
    λ (ECT coef)    : -0.654
    SE              : 0.178
    95% CI          : [-1.006, -0.302]
    p-value         : 0.001
    |λ| < 1?        : Yes  ✓
  
  Outlier Dummy Coefficients:
    Outlier_2020Q2:     2.345  (p=0.012)  ← Q2 2020 was 2.3 pp higher
    Outlier_2021Q2:     1.876  (p=0.034)  ← Q2 2021 was 1.9 pp higher
    Outlier_2022Q2:     0.123  (p=0.756)  ← Not significant

  Interpretation:
    - Adding outlier dummies reduces |λ| from 1.087 → 0.654
    - 2020-Q2 and 2021-Q2 were major outlier quarters
    - Speed-of-adjustment is stable once outliers controlled
    - Suggests |λ|>1 was driven by crisis-period shocks

Step 3B Output Example: Huber Robust Regression
  ──────────────────────────────────────────────

  ECM with Huber Robust Regression (M-estimator):
    λ (ECT coef)    : -0.681
    SE              : 0.162
    95% CI          : [-0.999, -0.363]
    p-value         : 0.001
    |λ| < 1?        : Yes  ✓
  
  Comparison with OLS:
    OLS λ           : -0.712
    Difference (Huber - OLS): 0.031

  Interpretation:
    - Huber robust regression gives λ = -0.681 (very similar to OLS -0.712)
    - Small difference indicates outliers not extreme, but present
    - Both methods support stable error correction (|λ| < 1)

Step 4 Output Example: DOLS Alternative
  ──────────────────────────────────────

  DOLS Long-Run Coefficients:
    GDP_Growth_Rate: -0.1489  (SE=0.0234, p=0.000)
    Inflation_Rate:  -0.0801  (SE=0.0156, p=0.001)
    Youth_LFPR:      -0.1156  (SE=0.0345, p=0.002)
    Remittances_USD:  0.0007  (SE=0.0002, p=0.004)
  
  ECM with DOLS-Based ECT:
    λ (ECT_DOLS coef) : -0.598
    SE                : 0.171
    95% CI            : [-0.935, -0.261]
    p-value           : 0.001
    |λ| < 1?          : Yes  ✓

  Interpretation:
    - DOLS coefficients very similar to baseline ARDL
    - DOLS-based ECT yields λ = -0.598 (also < 1)
    - Supports robustness of long-run relationships

Step 5A Output Example: CUSUM Stability Test
  ─────────────────────────────────────────

  CUSUM: No structural breaks detected (within 95% bounds)
  CUSUM-SQ: No structural breaks detected (within 95% bounds)

  Files generated:
    - cusum_stability_test.png (visual plot)

  Interpretation:
    - Recursive residuals remain within confidence bounds
    - No evidence of breaks in ECM short-run dynamics
    - Model appears stable over entire sample period

Step 5B Output Example: Recursive λ Estimation
  ────────────────────────────────────────────

  Recursive λ Summary:
    Initial λ (window n=20): -1.234
    Final λ (full sample):   -0.654
    Min λ: -1.456 at 2022Q1
    Max λ: -0.234 at 2024Q4
    Std(λ): 0.451

  Files generated:
    - recursive_lambda_estimate.png (visual plot)

  Interpretation:
    - λ starts > −1 (unstable), becomes < −1 during crisis, recovers to < 1
    - This pattern suggests |λ|>1 is temporary, driven by crisis shocks
    - Recent quarters (2024Q4) show |λ| < 1 clearly
    - Model has converged to stability after crisis period

Step 6 Output: Summary Table
  ──────────────────────────

  Specification              λ       SE       95% CI              |λ|<1?
  ────────────────────────────────────────────────────────────────────
  Baseline ARDL(3,1,1,2,0)  -1.087  0.236   [-1.550, -0.624]    No
  BIC-optimal: ARDL(1,0,1,1,0) -0.456  0.142   [-0.735, -0.177]    Yes ✓
  + Outlier Dummies (OLS)   -0.654  0.178   [-1.006, -0.302]    Yes ✓
  Huber Robust ECM          -0.681  0.162   [-0.999, -0.363]    Yes ✓
  DOLS-Based ECT            -0.598  0.171   [-0.935, -0.261]    Yes ✓

  Academic Paragraph Generated:
    [See robustness_paragraph.txt]

═══════════════════════════════════════════════════════════════════════════════
6. INTERPRETATION GUIDE: WHAT DOES EACH DIAGNOSTIC TELL YOU?
═══════════════════════════════════════════════════════════════════════════════

Baseline Problem:
  ─────────────

  If |λ| > 1, it implies:
    - System overshoots equilibrium by >100% per quarter
    - Economically implausible (would oscillate violently)
    - Suggests model misspecification
  
  Root causes could be:
    1. Too many lags → overfitting → unstable AR dynamics
    2. Multicollinearity → spurious AR coefficients
    3. Outliers → dominating residuals, inflating errors
    4. Sample size too small → degrees of freedom problem
    5. True economic process is I(2) → breaks cointegration assumption

Step 1 (Lag Re-selection) Diagnosis:
  ──────────────────────────────

  If BIC-optimal order achieves |λ| < 1:
    → GOOD: Problem likely due to over-parameterization in baseline
    → ACTION: Use BIC-optimal order as preferred specification
  
  If BIC-optimal also has |λ| > 1:
    → Problem deeper; proceed to Steps 2, 3, 4

Step 2 (Multicollinearity) Diagnosis:
  ────────────────────────────────

  If max VIF < 10:
    → GOOD: No severe multicollinearity
    → Proceed to other diagnostics
  
  If max VIF > 10:
    → HIGH: Multicollinearity is amplifying AR lag coefficients
    → ACTION: Use Ridge regularized coefficients
    → After Ridge regularization, check if |λ| < 1

Step 3 (Outliers) Diagnosis:
  ───────────────────────

  If λ improves substantially with outlier dummies:
    → GOOD: Outliers were distorting error correction
    → ACTION: Use outlier-controlled specification in paper
    → Explain outliers as crisis-driven temporary shocks
  
  If Huber robust regression differs little from OLS:
    → GOOD: Model is robust to outlier downweighting
    → Outliers not extreme; OLS estimates are reliable

Step 4 (DOLS) Diagnosis:
  ──────────────────

  If DOLS long-run coefficients differ from ARDL:
    → Suggests simultaneity bias in ARDL
    → DOLS is more reliable for long-run relationship
    → Use DOLS coefficients if λ_DOLS < 1
  
  If DOLS and ARDL very similar:
    → Timing assumptions in ARDL are OK
    → ARDL is valid estimator for long-run

Step 5A (CUSUM) Diagnosis:
  ─────────────────────

  If CUSUM within bounds entire period:
    → GOOD: No structural breaks in short-run dynamics
    → Model is stable
  
  If CUSUM breaches bounds around crisis quarters:
    → Crisis caused temporary break in ECM dynamics
    → Supports using outlier dummies or robust regression

Step 5B (Recursive λ) Diagnosis:
  ────────────────────────────

  If λ starts < 1, stays < 1:
    → Model was stable all along; overshooting may be spurious
    → Check matrix calculations; recalculate from scratch
  
  If λ starts < 1, becomes ≤ −1 at crisis, recovers:
    → Crisis years caused temporary instability
    → Model has converged to stability after crisis
    → Standard outlier controls should be sufficient
  
  If λ always < −1, never recovers:
    → Deeper structural change in economic relationships
    → Consider splitting sample or allowing time-varying parameters
    → May indicate shift from old to new equilibrium

Step 5C (Chow) Diagnosis:
  ──────────────────────

  If Chow p-value > 0.05:
    → No significant structural break at 2022-Q1
    → Model parameters are stable across periods
  
  If Chow p-value < 0.05:
    → Structural break detected at 2022-Q1 (firm economic crisis)
    → Justify using robust methods to handle break

═══════════════════════════════════════════════════════════════════════════════
TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════════════════

Error: KeyError: 'ECT' in formula
  ─────────────────────────────────

  Cause: ECT variable not in DataFrame (NaN values?)
  Fix:
    - Check create_ect_variables() function returns valid ECT
    - Verify lr_coefs and alpha_lr are correctly extracted from baseline
    - Check data has no gaps (use df.asfreq("QS") to fill quarterly)

Error: ValueError: less observations than variables
  ────────────────────────────────────────────────

  Cause: Sample is too short for ECM specification
  Fix:
    - Use simpler ECM (fewer lags of ΔU)
    - Increase n_start in recursive_lambda_estimation()
    - Check data has correct frequency (quarterly)

Error: Ridge regression not converging
  ────────────────────────────────────

  Cause: Extreme multicollinearity; Ridge penalty too small
  Fix:
    - Increase Ridge alphas range: np.logspace(-2, 4, 200)
    - Check if predictors are trending (differencing may help)
    - Try principal components regression instead

Error: Huber robust regression warnings
  ────────────────────────────────────

  Cause: Convergence issues in M-estimation
  Fix:
    - This is typically just a warning; results are still valid
    - Try adjusting Huber constant: HuberT(t=1.345) to HuberT(t=1.5)

Grid search taking too long (>10 minutes)
  ────────────────────────────────────────

  Cause: Search space is too large
  Fix:
    - Reduce p_max to 2 or 1
    - Reduce q_max to 1
    - Call: grid_df, best = lag_order_grid_search(df, p_max=2, q_max=1)

═══════════════════════════════════════════════════════════════════════════════

For questions or issues, refer to:
  - statsmodels documentation: https://www.statsmodels.org/
  - ARDL bounds testing: Pesaran, Shin & Smith (2001)
  - ECM theory: Engle & Granger (1987)

"""

if __name__ == "__main__":
    print(__doc__)
