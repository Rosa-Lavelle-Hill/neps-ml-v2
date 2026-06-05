# NEPS ML v2

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

For this repository, we keep the full `Results/runs/2026-04-14_083401__*/` family only: `__all`, `__prior_reading__student`, `__prior_reading__student__parent`, and `__prior_reading__student__teacher`. That is the most recent complete run group in the workspace and it contains the key .csv outputs and plot files readers need without preserving the whole run history.

Kept as Supplementary Online Resources for the Reader:

1. `Outputs/preprocessing_config.yaml` and `Outputs/pre_processing_output.txt` so readers can see the exact preprocessing settings and a run log.
2. `Outputs/dropped_variables_*.csv` and `Outputs/IV_correlations/*dropped_vars*.csv` so readers can inspect which variables were removed and why.
3. `Outputs/IV_correlations/*before*` and `Outputs/IV_correlations/*after*` correlation tables so readers can see the multicollinearity filtering step.
4. `Outputs/Phi_values/phi_matrix_before_changes.csv` and `Outputs/Phi_values/phi_pairs_above_0.7_before.csv` for binary-variable association checks.
5. `Outputs/mutual_information/mutual_information_scores.csv` if mutual information was actually computed for the final run.
6. `Outputs/category_distributions/X_train_counts.txt`, `Outputs/category_distributions/X_test_counts.txt`, and `Outputs/category_distributions/X_counts.txt` to document the category-count screening.
7. `Results/runs/2026-04-14_083401__*/Prediction/All_Models/all_test_scores_2026-04-14_083401__*_test.csv` and `Results/runs/2026-04-14_083401__*/Prediction/All_Models/all_model_errors_2026-04-14_083401__*_test.csv` for the core model comparison results.
8. `Results/runs/2026-04-14_083401__*/Interpretation/Permutation/*.csv` and `Results/runs/2026-04-14_083401__*/Interpretation/SHAP/*.csv` for the interpretability outputs.
9. `Results/runs/2026-04-14_083401__*/Interpretation/Permutation/Plots/*.png`, `Results/runs/2026-04-14_083401__*/Interpretation/Permutation/Grouped/Plots/*.png`, `Results/runs/2026-04-14_083401__*/Interpretation/SHAP/Plots/*.png`, and `Results/runs/2026-04-14_083401__*/Interpretation/SHAP/Grouped/Plots/*.png` for figure outputs.
10. `Results/runs/2026-04-14_083401__*/Prediction/Plots/*predicted_actual*.png` for model prediction plots.
11. `Results/runs/2026-04-14_083401__*/Prediction/All_Models/Plots/*.png` for aggregate model comparison figures (e.g., R2, RMSE, MAE summary plots).
12. `Results/runs/2026-04-14_083401__*/Prediction/*.txt` for model run log summaries (one per model per subset run).
13. `Results/runs/2026-04-14_083401__*/Prediction/Best_Params/*` for saved best hyperparameter files.
14. `Results/runs/2026-04-14_083401__*/config.yaml` and `Results/runs/2026-04-14_083401__*/config.json` for run-level configuration provenance.
15. `Results/stat_tests/best_models_errors_*.csv` for the statistical comparison inputs used after model fitting.

Suggested mapping from paper output to file:

1. Model comparison tables: `Results/runs/2026-04-14_083401__*/Prediction/All_Models/all_test_scores_2026-04-14_083401__*_test.csv`.
2. Per-model residual/error summaries: `Results/runs/2026-04-14_083401__*/Prediction/All_Models/all_model_errors_2026-04-14_083401__*_test.csv`.
3. Permutation importance and SHAP summaries: `Results/runs/2026-04-14_083401__*/Interpretation/Permutation/*.csv` and `Results/runs/2026-04-14_083401__*/Interpretation/SHAP/*.csv`.
4. Figure panels: `Results/runs/2026-04-14_083401__*/Interpretation/Permutation/Plots/*.png`, `Results/runs/2026-04-14_083401__*/Interpretation/Permutation/Grouped/Plots/*.png`, `Results/runs/2026-04-14_083401__*/Interpretation/SHAP/Plots/*.png`, `Results/runs/2026-04-14_083401__*/Interpretation/SHAP/Grouped/Plots/*.png`, `Results/runs/2026-04-14_083401__*/Prediction/Plots/*.png`, and `Results/runs/2026-04-14_083401__*/Prediction/All_Models/Plots/*.png`.

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

The repository currently uses ignored output folders so that the working tree stays small. The `.gitignore` files are tuned to keep only the latest canonical run group in `Results/runs/2026-04-14_083401__*/` with selected .csv and plot outputs, while still ignoring older run history artifacts and split data files.