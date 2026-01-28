import pandas as pd
import numpy as np
from pathlib import Path
from Functions.stat_tests import compare_models 

# Settings
test = True # If True, extract data from a test run <-- only to check that the code works
run_ids = ["2026-01-28_140927"] # <- might need to be a list of different IDs if comparing multiple runs not ran in parellel
subsets = ["prior_reading__student", "prior_reading__student__parent", "prior_reading__student__teacher", "all"]
best_from_ml_models_only = True  # If True, only consider ML models (DT, RF, HGB, XGB) when selecting best model per subset

# ================== Extract best models errors from runs ==========================
if test == True:
    run = "_test"
else:
    run = ""

if len(run_ids) == 1:
    run_ids = run_ids * len(subsets)    

best_models_errors = {}
for counter, (run_id, subset) in enumerate(zip(run_ids, subsets)):
    print(f"Processing run: {run_id}, subset: {subset}")
    csv_path = Path(f"Results/runs/{run_id}__{subset}/Prediction/All_Models/all_model_errors_{run_id}__{subset}{run}.csv")   

    # load data
    df = pd.read_csv(csv_path, index_col=0)
    if counter == 0:
        best_models_errors["Prior Reading"] = df["Prior Ach.+ Track"]    #<- later, might want to switch to just prior achievement

    # extract name of best model
    r2_all_models = pd.read_csv(Path(f"Results/runs/{run_id}__{subset}/Prediction/All_Models/all_test_scores_{run_id}__{subset}{run}.csv"), index_col=0)
    print(r2_all_models)
    if best_from_ml_models_only == True:
        r2_all_models = r2_all_models[["DT", "RF", "HGB", "XGB"]]
    r2_series = r2_all_models.loc["R2"]


    best_model = r2_series.idxmax()
    best_r2 = r2_series.max()
    print(f"Best model by R²: {best_model} (R² = {best_r2:.3f})")

    # extract errors of best model
    errors = df[best_model].values

    # save errors to new csv
    if subset == "all":
        subset_name = "Prior Reading + Student + Parent + Teacher"
    else:
        subset_name = subset.replace("__", " + ").replace("_", " ").title()
    best_models_errors[subset_name] = errors

# save all best models errors to csv
errors_df = pd.DataFrame.from_dict(best_models_errors)
errors_save_path = Path(f"Results/stat_tests/best_models_errors_{run_id}{run}.csv")
errors_df.to_csv(errors_save_path)
print(f"Saved best models errors to: {errors_save_path}")
#=======================================================================================
# Calculate a paired t-test between the absolute models errors (using only the best model per subset)
# 1.1. Prior Reading (M1) vs. Prior Reading + Student (M2)
results = compare_models(
    df=errors_df,
    model_a="Prior Reading",
    model_b="Prior Reading + Student",
    alternative="two-sided"   
)
print("Comparing model A (Prior Reading) vs. model B (Prior Reading + Student)")
for k, v in results.items():
    print(f"{k}: {v}")

# 2.1. Prior Reading + Student (M2) vs. Prior Reading + Student + Parent (M3)
results = compare_models(
    df=errors_df,
    model_a="Prior Reading + Student",
    model_b="Prior Reading + Student + Parent",
    alternative="two-sided"   
)   
print("Comparing model A (Prior Reading + Student) vs. model B (Prior Reading + Student + Parent)")
for k, v in results.items():
    print(f"{k}: {v}")  

# 2.2. Prior Reading + Student (M2) vs. Prior Reading + Student + Teacher (M4)
results = compare_models(
    df=errors_df,
    model_a="Prior Reading + Student",
    model_b="Prior Reading + Student + Teacher",
    alternative="two-sided"   
)   
print("Comparing model A (Prior Reading + Student) vs. model B (Prior Reading + Student + Teacher)")    
for k, v in results.items():
    print(f"{k}: {v}")  

# 2.3. Prior Reading + Student (M2) vs. All (M5)
results = compare_models(
    df=errors_df,
    model_a="Prior Reading + Student",
    model_b="Prior Reading + Student + Parent + Teacher",
    alternative="two-sided"   
)
print("Comparing model A (Prior Reading + Student) vs. model B (All Predictors)")
for k, v in results.items():
    print(f"{k}: {v}")

print("Statistical comparison complete.")
# ------------------------------------------------------------------
