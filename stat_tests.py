import pandas as pd
import numpy as np
from pathlib import Path
from Functions.stat_tests import compare_models 

# Settings
test = True
run_ids = ["2026-01-28_140927"] # <- might need to be a list if comparing multiple runs
subsets = ["prior_reading__student", "prior_reading_student_parent", "prior_reading__student_teacher", "all"]

# ================== Extract best models errors from runs ==========================
if test == True:
    run = "_test"
else:
    run = ""

best_models_errors = {}
for counter, (run_id, subset) in enumerate(zip(run_ids, subsets)):
    print(f"Processing run: {run}, subset: {subset}")
    csv_path = Path(f"Results/runs/{run}__{subset}/Prediction/All_Models/all_model_errors_{run_id}{run}.csv")   

    # load data
    df = pd.read_csv(csv_path, index_col=0)
    if counter == 0:
        best_models_errors["Prior Reading"] = df["Prior Ach. + Track"]    #<- later, might want to switch to just prior achievement

    # extract name of best model
    r2_all_models = pd.read_csv(Path(f"Results/runs/{run}__{subset}/Prediction/All_Models/all_test_scores_{run_id}{run}.csv"), index_col=0)
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
        subset_name = subset.replace("__", " + ").replace("_", " ")
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

for k, v in results.items():
    print(f"{k}: {v}")

# 2.1. Prior Reading + Student (M2) vs. Prior Reading + Student + Parent (M3)
results = compare_models(
    df=errors_df,
    model_a="Prior Reading + Student",
    model_b="Prior Reading + Student + Parent",
    alternative="two-sided"   
)   

for k, v in results.items():
    print(f"{k}: {v}")  

# 2.2. Prior Reading + Student (M2) vs. Prior Reading + Student + Teacher (M4)
results = compare_models(
    df=errors_df,
    model_a="Prior Reading + Student",
    model_b="Prior Reading + Student + Teacher",
    alternative="two-sided"   
)   

for k, v in results.items():
    print(f"{k}: {v}")  

# 2.3. Prior Reading + Student (M2) vs. All (M5)
results = compare_models(
    df=errors_df,
    model_a="Prior Reading + Student",
    model_b="Prior Reading + Student + Parent + Teacher",
    alternative="two-sided"   
)

for k, v in results.items():
    print(f"{k}: {v}")

print("Statistical comparison complete.")
# ------------------------------------------------------------------
