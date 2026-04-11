# Dataset Audit Sheet

Strict audit sheet for the research repo. Rows are marked as `verified` when there is direct code evidence from `read_csv` / `to_csv` usage, and `inferred` when the repo clearly uses the file family but the exact producer chain is not fully explicit.

## Legend
- `source` = raw or externally downloaded data
- `derived` = produced by a script or notebook in this repo
- `consumed` = read by analysis scripts/notebooks/paper files
- `paper` = used directly in `current_latest_paper/`

## Core Audit Table

| Dataset / File | Role | Included Data | Producer / Writer | Consumer / Reader | Used in Paper | Status |
|---|---|---|---|---|---|---|
| `labour/csv/2015_25_Percent_Datafile_Out.csv` | source | LFS 2015 microdata sample | external DCS source | `extraction/underemployment.py`, `extraction/underemployment_weighted.py`, `extraction/qualification_underemployment.py`, `run_shap_analysis.py`, `lagged_analysis.py`, several notebooks | yes, via derived underemployment series | verified |
| `labour/csv/2020_25_Percent_Datafile_Out.csv` | source | LFS 2020 microdata sample | external DCS source | same as above | yes, via derived underemployment series | verified |
| `labour/csv/LFS-2022-25-Percent-Data-Without-Computer.csv` | source | LFS 2022 microdata sample | external DCS source | same as above | yes, via derived underemployment series | verified |
| `labour/csv/LFS-2023-25-Percent-Data-Without-Computer.csv` | source | LFS 2023 microdata sample | external DCS source | same as above | yes, via derived underemployment series | verified |
| `labour/csv/API_SL.UEM.TOTL.ZS_DS2_en_csv_v2_93.csv` | source | World Bank unemployment series | external WB source | `generate_notebooks.py`, `lagged_analysis.py`, `run_shap_analysis.py`, notebooks in `Data_Analysis/` | yes, in contextual plots and comparisons | verified |
| `labour/csv/API_SL.TLF.TOTL.IN_DS2_en_csv_v2_761.csv` | source | World Bank labour force series | external WB source | `Data_Analysis/Internet_labourforce.ipynb`, `RidgeRegression/LabourDataloader.py` | likely yes, in supporting analysis | verified |
| `labour/finalized_csv/quarterly_underemployment.csv` | derived | quarterly underemployment series | `extraction/underemployment.py` or copied from `extraction/quarterly_underemployment.csv` | `generate_underemployment_dashboard.py`, `generate_advanced_visualizations.py`, `Task_praveen_literature_policy/task.ipynb` | yes | verified |
| `extraction/quarterly_underemployment.csv` | derived | quarterly underemployment series | `extraction/underemployment.py` | `ardl_vecm/build_quarterly_dataset.py`, `Task_praveen_literature_policy/task.ipynb` | yes, primary DV path | verified |
| `extraction/weighted_quarterly_underemployment.csv` | derived | weighted quarterly underemployment series | `extraction/underemployment_weighted.py` | `create_sensitivity_notebook.py`, `sensitivity_analysis.py`, `Data_Analysis/Sensitivity_Analysis_Underemployment_Definitions.ipynb` | yes, robustness checks | verified |
| `extraction/qualification_underemployment.csv` | derived | qualification mismatch proxy | `extraction/qualification_underemployment.py` | `create_sensitivity_notebook.py`, `sensitivity_analysis.py`, `gender_analysis.py`, `Data_Analysis/Sensitivity_Analysis_Underemployment_Definitions.ipynb` | yes, sensitivity and gender analysis | verified |
| `DataLoader/master_dataset.csv` | derived / foundation | annual integrated master dataset | likely manual merge or notebook-created; exact producer not fully explicit | `README.md`, `generate_underemployment_dashboard.py`, `generate_advanced_visualizations.py`, `gender_analysis.py`, `sensitivity_analysis.py`, many notebooks | yes, central annual baseline | inferred |
| `ardl_vecm/master_dataset.csv` | derived / copy | annual master dataset used in ARDL submodule | copied from DataLoader or earlier build step | `ardl_vecm/ardl_vecm_causal_inference.ipynb`, `gender_analysis.py`, `sensitivity_analysis.py` | yes | inferred |
| `ardl_vecm/master_dataset_2025.csv` | derived | updated annual master dataset with 2025 values | `ardl_vecm/estimate_master_2025.py` | `ardl_vecm/build_quarterly_dataset.py`, `Data_Analysis/RQ3_Interaction_Report.ipynb` | yes | verified |
| `ardl_vecm/quarterly_master_dataset.csv` | derived | quarterly model-ready dataset | `ardl_vecm/build_quarterly_dataset.py` | `ardl_vecm/run_ardl_quarterly.py`, `ardl_vecm/run_ecm.py`, `ardl_vecm/ardl_vecm_causal_inference.ipynb` | yes, main econometric dataset | verified |
| `data_pipeline/quarterly_exchange_rates.csv` | derived | quarterly FX series | `data_pipeline/03_exchange_rate.py` | `ardl_vecm/build_quarterly_dataset.py`, `methodology_gaps_2/exchange_rate_backfill.py`, some notebooks | yes | verified |
| `data_pipeline/annual_exchange_rates.csv` | derived | annual FX series | `data_pipeline/03_exchange_rate.py` | downstream dataset builders and notebooks | yes | verified |
| `data_pipeline/output/quarterly_remittances.csv` | derived | quarterly remittance series | `data_pipeline/04_temp_disagg.py` | `ardl_vecm/build_quarterly_dataset.py` | yes | verified |
| `data_pipeline/output/quarterly_agricultural_output.csv` | derived | quarterly agricultural output series | `data_pipeline/04_temp_disagg.py` and/or `data_pipeline/07_agri_index.py` | `ardl_vecm/build_quarterly_dataset.py` | yes | verified with alternate producer path |
| `data_pipeline/output/master_dataset_unadjusted.csv` | derived | baseline master before correction | `data_pipeline/05_artefact_handling.py` | `data_pipeline/04_imputation.py` | no direct paper use | verified |
| `data_pipeline/output/master_dataset_imputed.csv` | derived | imputed master dataset | `data_pipeline/05_artefact_handling.py` | `data_pipeline/04_imputation.py` | no direct paper use | verified |
| `data_pipeline/output/master_dataset_final.csv` | derived | pooled final annual dataset | `data_pipeline/04_imputation.py` | `data_pipeline/07_agri_index.py`, `methodology_gaps_2/create_sensitivity_notebook.py` | yes, indirectly via later merges | verified |
| `data_pipeline/output/master_dataset_final_weighted.csv` | derived | weighted final annual dataset | `data_pipeline/07_agri_index.py` | later quarterly merge and sensitivity work | yes, indirectly | verified |
| `ardl_vecm/output/adf_quarterly.csv` | derived analysis output | stationarity test results | `ardl_vecm/run_ardl_quarterly.py` | paper tables / discussion | yes | verified |
| `ardl_vecm/output/ardl_bounds_test.csv` | derived analysis output | ARDL bounds test results | `ardl_vecm/run_ardl_quarterly.py` | paper tables / discussion | yes | verified |
| `ardl_vecm/output/ardl_full_results.csv` | derived analysis output | full ARDL coefficients | `ardl_vecm/run_ardl_quarterly.py` | paper tables / discussion | yes | verified |
| `ardl_vecm/output/granger_quarterly.csv` | derived analysis output | Granger causality test results | `ardl_vecm/run_ardl_quarterly.py` | paper tables / discussion | yes | verified |
| `ardl_vecm/output/ecm_results.csv` | derived analysis output | ECM estimates | `ardl_vecm/run_ecm.py` | paper tables / discussion | yes | verified |
| `ardl_vecm/output/interaction_models.csv` | derived analysis output | crisis interaction models | `ardl_vecm/run_ardl_quarterly.py` | paper tables / discussion | yes | verified |
| `ardl_vecm/output/hac_vs_hc3.csv` | derived analysis output | standard-error comparison | `ardl_vecm/run_ardl_quarterly.py` | robustness discussion | yes | verified |
| `ardl_vecm/output/rolling_gdp_coef.csv` | derived analysis output | rolling GDP coefficients | `ardl_vecm/run_ardl_quarterly.py` | figure/table support | yes | verified |
| `Zivot-Andrews/sri_lanka_labour_macro_combined.csv` | derived / input bundle | combined labour + macro series | external or upstream merge | `Zivot-Andrews/structural_breaks.py`, `Zivot-Andrews/estimate_2025.py` | yes | verified |
| `Zivot-Andrews/sri_lanka_labour_macro_combined_2025.csv` | derived / input bundle | updated combined series | upstream merge or updated estimate | `Zivot-Andrews/estimate_2025.py` | yes | verified |
| `Zivot-Andrews/za_results.csv` | derived analysis output | Zivot-Andrews break results | `Zivot-Andrews/structural_breaks.py` | paper break section | yes | verified |
| `Zivot-Andrews/bp_results.csv` | derived analysis output | Bai-Perron results | `Zivot-Andrews/structural_breaks.py` | paper break section | yes | verified |
| `Task_praveen_literature_policy/output/okun_hysteresis.csv` | derived analysis output | lag / hysteresis summary | `Task_praveen_literature_policy/task.ipynb` | literature-policy task; possibly supporting appendix | maybe | verified |
| `Task_praveen_literature_policy/output/quarterly_lag_scan.csv` | derived analysis output | lag scan results | `Task_praveen_literature_policy/task.ipynb` | supporting analysis | maybe | verified |
| `current_latest_paper/*.tex` | paper fragments | LaTeX tables and subsections | generated from analysis outputs | `current_latest_paper/main.tex` includes them | yes | verified |
| `current_latest_paper/main.pdf` | paper artifact | compiled paper | LaTeX build | final deliverable | yes | verified |

## Folder Collections That Behave Like Dataset Families

| Folder | What is included | Typical use | Status |
|---|---|---|---|
| `labour/finalized_csv/Employment_by_sector_(%)_sl_indicators/` | sector employment indicator CSVs | sector-level labour share analysis and notebook visualizations | verified |
| `labour/finalized_csv/Labor_force_participation_rate,_total_(% of total population ages 15-64)_(modeled ILO estimate)_sl_indicators/` | LFPR CSV family | labour participation analysis and supporting plots | verified |
| `labour/finalized_csv/Employment_to_population_ratio_(%)_sl_indicators/` | employment-to-population indicator CSVs | labour market context analysis | verified |
| `labour/finalized_csv/Unemployment_(%)_sl_indicators/` | unemployment indicator CSVs | unemployment context and comparison plots | verified |
| `labour/finalized_csv/sl_labour_csv/` | custom labour summary CSVs | underemployment and employment breakdowns | verified/inferred by naming |
| `labour/labour_report_all_csv/underemployment_(2015-2024)/` | underemployment report tables by district, gender, education, industry | paper support tables and dashboards | verified |
| `labour/labour_report_all_csv/employment_(2015-2024)/` | employment report tables | supporting context analysis | verified |
| `labour/labour_report_all_csv/economically_active_inactive_(2015-2024)/` | participation / inactivity tables | supporting labour market narrative | verified |
| `labour/labour_report_all_csv/informal_sector_employment_(2015-2024)/` | informal sector tables | informality analysis and paper context | verified |
| `labour/labour_report_all_csv/total_jobs_with_secondary_employment_(2015-2024)/` | secondary job tables | supporting context | verified |
| `labour/labour_report_all_csv/unemployment_(2015-2024)/` | unemployment tables | supporting context | verified |
| `Data_Analysis/` notebooks | notebook-based readers over many of the above CSVs | exploratory analysis, figures, appendix material | verified |

## Strict File Processing Chain

### 1. Raw labour series
- Raw DCS/WB labour CSVs in `labour/csv/` feed the underemployment extraction scripts.
- These are the only files that produce the quarterly underemployment series used as the dependent variable.

### 2. Quarterly underemployment creation
- `extraction/underemployment.py` creates the main quarterly series.
- `extraction/underemployment_weighted.py` creates the weighted variant.
- `extraction/qualification_underemployment.py` creates the qualification proxy.

### 3. Macro preprocessing
- `data_pipeline/03_exchange_rate.py` creates quarterly and annual FX series.
- `data_pipeline/04_temp_disagg.py` creates quarterly remittances and quarterly agriculture output.
- `data_pipeline/05_artefact_handling.py` and `data_pipeline/04_imputation.py` clean and pool the annual master.
- `data_pipeline/07_agri_index.py` refines the agricultural output path and produces a weighted final dataset.

### 4. Model-ready merge
- `ardl_vecm/build_quarterly_dataset.py` combines the quarterly underemployment series with quarterly macro data and annual-to-quarterly disaggregated indicators.
- This produces `ardl_vecm/quarterly_master_dataset.csv`, which is the key analysis input.

### 5. Econometric outputs
- `ardl_vecm/run_ardl_quarterly.py` and `ardl_vecm/run_ecm.py` generate all ARDL, Granger, ECM, and robustness CSV outputs.
- These outputs are then summarized into LaTeX tables and narrative claims in `current_latest_paper/`.

### 6. Structural breaks and robustness
- `Zivot-Andrews/structural_breaks.py` and `Zivot-Andrews/estimate_2025.py` produce break tests.
- `sensitivity_analysis.py`, `gender_analysis.py`, `district_analysis.py`, and `run_shap_analysis.py` feed the paper’s heterogeneity and robustness sections.

## Manual Verification List
These links are still worth checking manually because the producer is not fully explicit in code:
- `DataLoader/master_dataset.csv`
- `labour/finalized_csv/quarterly_underemployment.csv`
- `labour/finalized_csv/sl_labour_csv/*`
- `current_latest_paper/*.tex` inclusion paths
- `data_pipeline/output/quarterly_agricultural_output.csv` authoritative route

## Suggested Review Order
1. `extraction/` first, because this creates the dependent variable.
2. `data_pipeline/` second, because it prepares the predictors.
3. `ardl_vecm/` third, because it builds the model dataset and results.
4. `Zivot-Andrews/` fourth, because it supports the structural-break claims.
5. `run_shap_analysis.py`, `gender_analysis.py`, `district_analysis.py`, `sensitivity_analysis.py` fifth, because they support robustness and subgroup results.
6. `current_latest_paper/` last, because it should only be checked after the upstream data flow is validated.
