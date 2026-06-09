# NEPS-ML v2

This repository contains the code and supplementary materials for the registered report **“A Registered Report (Stage 1): Using Machine Learning to Understand the Predictors of Reading Comprehension.”**
At this stage we are using a synthetic y variable to build the code base.

The codebase is organised as a reproducible pipeline:

1. `preprocessing.py` prepares the raw NEPS data and writes the derived preprocessing artifacts into `Outputs/`.
2. `run_master.py` loads the YAML configuration files in `configs/` and launches the modelling pipeline.
3. `ml_pipeline.py` trains and evaluates the models, writing run-specific results into `Results/runs/<run_id>/`.
4. The analysis and supplementary scripts (`stat_tests.py`, `conditional_importance_tests.py`, `group_correlations.py`, `plot_CIT.py`) reuse those saved outputs for downstream figures and statistical tests.

## Reproducing The Pipeline

The repository is designed so that the important settings are stored in versioned YAML files. The main entry points are:

1. `configs/base.yaml` for the modelling and global run settings.
2. `configs/preprocessing.yaml` for preprocessing thresholds and switches.
3. `configs/subsets/*.yaml` for the different predictor subsets used in the paper.

Typical execution order is:

1. Run preprocessing with `python preprocessing.py`.
2. Run the modelling suite with `python run_master.py`.
3. Run follow-up analyses such as `python stat_tests.py` or `python conditional_importance_tests.py` against a chosen `Results/runs/<run_id>/` folder.

The exact configuration used for a preprocessing run is copied to `Outputs/preprocessing_config.yaml`. The exact configuration used for each modelling run is saved to `Results/runs/<run_id>/config.yaml` and `Results/runs/<run_id>/config.json`.

## What Is On GitHub

For this repository, we keep the full `Results/runs/2026-06-05_142231__*/` family only: `__all`, `__prior_reading__student`, `__prior_reading__student__home`, and `__prior_reading__student__pedagogical`. That is the most recent complete run group in the workspace and it contains the key .csv outputs and plot files readers need without preserving the whole run history.

Supplementary Online Resources for the Reader:

1. **Variable infomation:** see `Data/Meta/variable_info_agreed_Feb_2_2026.csv` for the information we used on the raw variables in preprocessing and `Data/Meta/var_info_used_in_model_final_groups.csv` for only the variables used in the final model (including created variables) and their groupings.
2. **Preprocessing settings and run logs:** see `Outputs/preprocessing_config.yaml` and `Outputs/pre_processing_output.txt`.
3. **Variables removed during preprocessing:** see `Outputs/dropped_variables_*.csv` and `Outputs/IV_correlations/*dropped_vars*.csv`.
4. **Correlation screening before and after filtering:** see `Outputs/IV_correlations/*before*` and `Outputs/IV_correlations/*after*`.
5. **Binary-variable association checks (phi):** see `Outputs/Phi_values/phi_matrix_before_changes.csv` and `Outputs/Phi_values/phi_pairs_above_0.7_before.csv`.
6. **Mutual information summary:** see `Outputs/mutual_information/mutual_information_scores.csv`.
7. **Category-count diagnostics:** see `Outputs/category_distributions/X_train_counts.txt`, `Outputs/category_distributions/X_test_counts.txt`, and `Outputs/category_distributions/X_counts.txt`.
8. **Main model performance tables:** see `Results/runs/2026-06-05_142231__*/Prediction/All_Models/all_test_scores_2026-06-05_142231__*_test.csv` and `Results/runs/2026-06-05_142231__*/Prediction/All_Models/all_model_errors_2026-06-05_142231__*_test.csv`.
9. **Permutation and SHAP output tables:** see `Results/runs/2026-06-05_142231__*/Interpretation/Permutation/*.csv` and `Results/runs/2026-06-05_142231__*/Interpretation/SHAP/*.csv`.
10. **Interpretation figures (permutation and SHAP):** see `Results/runs/2026-06-05_142231__*/Interpretation/Permutation/Plots/*.png`, `Results/runs/2026-06-05_142231__*/Interpretation/Permutation/Grouped/Plots/*.png`, `Results/runs/2026-06-05_142231__*/Interpretation/SHAP/Plots/*.png`, and `Results/runs/2026-06-05_142231__*/Interpretation/SHAP/Grouped/Plots/*.png`.
11. **Model prediction plots (predicted vs observed):** see `Results/runs/2026-06-05_142231__*/Prediction/Plots/*predicted_actual*.png`.
12. **Aggregate model comparison plots:** see `Results/runs/2026-06-05_142231__*/Prediction/All_Models/Plots/*.png` (e.g., R2, RMSE, MAE summaries).
13. **Per-model run log files:** see `Results/runs/2026-06-05_142231__*/Prediction/*.txt`.
14. **Best hyperparameter files:** see `Results/runs/2026-06-05_142231__*/Prediction/Best_Params/*`.
15. **Run-level configuration provenance:** see `Results/runs/2026-06-05_142231__*/config.yaml` and `Results/runs/2026-06-05_142231__*/config.json`.
16. **Statistical-comparison input table:** see `Results/stat_tests/best_models_errors_*.csv`.

<!-- Suggested mapping from paper output to file:

1. Model comparison tables: `Results/runs/2026-06-05_142231__*/Prediction/All_Models/all_test_scores_2026-06-05_142231__*_test.csv`.
2. Per-model residual/error summaries: `Results/runs/2026-06-05_142231__*/Prediction/All_Models/all_model_errors_2026-06-05_142231__*_test.csv`.
3. Permutation importance and SHAP summaries: `Results/runs/2026-06-05_142231__*/Interpretation/Permutation/*.csv` and `Results/runs/2026-06-05_142231__*/Interpretation/SHAP/*.csv`.
4. Figure panels: `Results/runs/2026-06-05_142231__*/Interpretation/Permutation/Plots/*.png`, `Results/runs/2026-06-05_142231__*/Interpretation/Permutation/Grouped/Plots/*.png`, `Results/runs/2026-06-05_142231__*/Interpretation/SHAP/Plots/*.png`, `Results/runs/2026-06-05_142231__*/Interpretation/SHAP/Grouped/Plots/*.png`, `Results/runs/2026-06-05_142231__*/Prediction/Plots/*.png`, and `Results/runs/2026-06-05_142231__*/Prediction/All_Models/Plots/*.png`. -->

## What Is Not Tracking In GitHub:

1. Older run folders under `Results/runs/`.
2. Full per-fold prediction dumps and scratch outputs.
3. Raw or preprocessed data. The raw data are available to researchers after registration via the NEPS website (https://www.neps-data.de).

## Repository Layout For Supplementary Material

1. `configs/` for all run settings.
2. `Outputs/` for preprocessing diagnostics and variable-screening artifacts.
3. `Results/runs/` for model-specific outputs.
4. `Results/stat_tests/` for significance testing between model families.
5. `Sim-CIT/` for conditional importance test inputs and outputs.

## Notes

The repository currently uses ignored output folders so that the working tree stays small. The `.gitignore` files are tuned to keep only the latest canonical run group in `Results/runs/2026-06-05_142231__*/` with selected .csv and plot outputs, while still ignoring older run history artifacts and split data files.