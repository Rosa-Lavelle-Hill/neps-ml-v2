# just use as a trial to see if it works
from datetime import datetime as dt
from pathlib import Path
import json
import joblib
import random
import yaml
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from Functions.CIT import diagnose_degenerate_features
from Functions.pipeline import construct_pipelines_all_no_imputation
from Functions.preprocessing_functions import drop_cols, remove_variables
from fixed_params import remove_vars
from Functions.subsets import cfg_get
from tofi import CIT


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def deep_merge(a, b):
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ==============================
# Script controls / run settings
# ==============================
# Base config + optional CIT override config.
cfg_base = load_yaml("configs/base.yaml")
cfg_override_path = Path("configs/conditional_importance.yaml")
cfg_override = load_yaml(cfg_override_path) if cfg_override_path.exists() else {}
cfg = deep_merge(cfg_base, cfg_override)

seed = cfg_get(cfg, ["params", "seed"])
test_size = cfg_get(cfg, ["modelling", "test_size"])
var_info_sheet = cfg_get(cfg, ["paths", "var_info_csv"])
preprocessed_data_path = cfg_get(cfg, ["paths", "preprocessed_data"])
categorical_features_csv = cfg_get(cfg, ["paths", "categorical_features_csv"])

cit_cfg = cfg.get("conditional_importance", {})

test = cit_cfg.get("test", "CPI") # Type of test conducted "RPT" or "CPI"
run_CIT = cit_cfg.get("run_CIT", True) # if False will skip CIT and just run diagnostics on 0 SE features; if True will run CIT and then diagnostics on 0 SE features
sample_features = cit_cfg.get("sample_features", False) # set to true if want to reduce computation time to test code -- will only run for a sample of features
run_diagnostics = cit_cfg.get("run_diagnostics", True) # whether to run diagnostics on 0 SE features after CIT; can set to False to skip diagnostics and just get CIT results
n_iter = cit_cfg.get("n_iter", 1000) # number of iterations for null distribution; only used if test == "RPT" or if test == "CPI" and null_dist == "permutation"; ignored if test == "CPI" and null_dist == "normality"
start_string = cit_cfg.get("run_id", "2026-04-14_083401__all") # must match the run folder id used by ml_pipeline.py under Results/runs/<run_id>/
test_run = cit_cfg.get("test_run", cfg.get("test_run_params", {}).get("test_run", True)) # controls ONLY the params filename suffix ("_test"); it does NOT modify run_id/path
model_name = cit_cfg.get("model_name", "CAT") # model_names = ["DT", "RF", "HGB", "XGB", "CAT"]
handle_missing = cit_cfg.get("handle_missing", False) # sampler choice in CIT only: False -> RandomForest sampler, True -> HistGradientBoosting sampler

# Missing-value handling strategy used across train/test preparation.
# Options:
# - "drop":   drop rows with any missing value in X or y
# - "impute": fill missing values with -999 (categoricals get sentinel category)
# NOTE: this controls learner/CIT input prep, not sampler class choice.
missing_value_strategy = cit_cfg.get("missing_value_strategy", "drop")

if sample_features == True:
    n_features = cit_cfg.get("n_features", 10) # number of features to sample
if test_run == True:
    t = '_test'
else:
    t = ''

start_time = dt.now()
save_path = Path(cit_cfg.get("save_root", "Sim-CIT"))

# NOTE:
# `start_string` should exactly match the run folder id used by ml_pipeline.py.
# Best params are saved at: Results/runs/<run_id>/Prediction/Best_Params/
# We reuse tuned hyperparameters from that run for the selected `model_name`.
run_id = start_string
params_path = Path("Results") / "runs" / run_id / "Prediction" / "Best_Params"

# ======================
# Load data + metadata
# ======================
# Import variable information & meta data:
var_info = pd.read_csv(var_info_sheet, encoding="utf-8", sep=None, engine="python")
var_info.columns = var_info.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
var_info = var_info[var_info["include as predictor"] == 1]
var_names_dict = dict(zip(var_info['var'], var_info['varname']))

# get block information for grouped importance
groups_info_path = Path("Data/Meta/var_info_used_in_model_final_groups.csv")
groups_info = pd.read_csv(groups_info_path, encoding="utf-8", sep=None, engine="python")
groups_info.columns = groups_info.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

required_group_cols = ["var", "Final Groups"]
missing_group_cols = [c for c in required_group_cols if c not in groups_info.columns]
if missing_group_cols:
    raise KeyError(
        f"Missing required columns {missing_group_cols} in {groups_info_path}. "
        f"Available columns: {list(groups_info.columns)}"
    )

groups_info = groups_info[groups_info["include as predictor"] == 1].copy()
block_dict = dict(zip(groups_info["var"], groups_info["Final Groups"].fillna("Unmapped").astype(str)))

print("Loading data...")
X_and_y = pd.read_csv(preprocessed_data_path, index_col=[0])

X = X_and_y.drop("y", axis=1)
print("X shape: " + str(X.shape))
y = X_and_y["y"]
print("y shape: " + str(y.shape))

# Ensure categorical features actually exist in X
categorical_features_df = pd.read_csv(categorical_features_csv, index_col=[0])
categorical_features_col = categorical_features_df.columns[0]
categorical_features_list = categorical_features_df[categorical_features_col].dropna().astype(str).tolist()
categorical_features = [c for c in categorical_features_list if c in X.columns]
print(len(categorical_features), "categorical features found in data.")

# Get numeric dataframe and feature names by dropping categorical columns
numerical_df, numerical_features = drop_cols(categorical_features, X)

# Avoid SettingWithCopyWarning
X = X.copy()

# Assign data types
X[categorical_features] = X[categorical_features].astype('category')
print(X[categorical_features])
X[numerical_features] = X[numerical_features].astype('float64')

# check all drop vars are removed
X = remove_variables(X, remove_vars)

# Check category counts across train/test data
# Lists are filtered after variable dropping so only valid columns remain.
numerical_features_in_data = [elem for elem in numerical_features if elem in list(X.columns)]
print("Data contains {} numerical features".format(len(numerical_features_in_data)))
categorical_features_in_data = [elem for elem in categorical_features if elem in list(X.columns)]
print("Data contains {} categorical features".format(len(categorical_features_in_data)))

numeric_features_index = X[numerical_features_in_data].columns
categorical_features_index = X[categorical_features_in_data].columns

# Construct all pipelines with no imputation
print("Constructing pipelines...")
pipe_dt, pipe_rf, pipe_hgb, pipe_xgb, pipe_cat = construct_pipelines_all_no_imputation(categorical_features_index)

# Select which fitted pipeline is evaluated inside CIT.
if model_name == "DT":
    pipe = pipe_dt
elif model_name == "RF":
    pipe = pipe_rf
elif model_name == "HGB":
    pipe = pipe_hgb
elif model_name == "XGB":
    pipe = pipe_xgb
elif model_name == "CAT":
    pipe = pipe_cat
else:
    raise ValueError(f"Unsupported model_name: {model_name}. Choose from ['DT','RF','HGB','XGB','CAT']")

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=seed, test_size=test_size, shuffle=True)
print("X_train shape: " + str(X_train.shape))
print("X_test shape: " + str(X_test.shape))

# Apply missing-value strategy (single control point for this script).
if missing_value_strategy not in ["drop", "impute"]:
    raise ValueError("missing_value_strategy must be either 'drop' or 'impute'.")

if missing_value_strategy == "drop":
    # Strict complete-case analysis: removes rows with NA in X or y.
    # Use this if you prefer not to introduce sentinel values.
    train_valid = y_train.notna() & ~X_train.isna().any(axis=1)
    test_valid = y_test.notna() & ~X_test.isna().any(axis=1)

    dropped_train = int((~train_valid).sum())
    dropped_test = int((~test_valid).sum())
    if dropped_train > 0:
        print(f"Dropping {dropped_train} train rows due to missing values in X or y.")
    if dropped_test > 0:
        print(f"Dropping {dropped_test} test rows due to missing values in X or y.")

    X_train = X_train.loc[train_valid].copy()
    y_train = y_train.loc[train_valid].copy()
    X_test = X_test.loc[test_valid].copy()
    y_test = y_test.loc[test_valid].copy()

    if X_train.empty or X_test.empty:
        print(
            "Complete-case dropping produced an empty split. "
            "Falling back to sentinel imputation (-999)."
        )

        # Recreate original split and impute so model fitting can proceed.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=seed, test_size=test_size, shuffle=True
        )

        cat_cols = X_train.select_dtypes(include='category').columns
        num_cols = X_train.select_dtypes(exclude='category').columns

        for col in cat_cols:
            cat_dtype = X_train[col].cat.categories.dtype
            sentinel = -999 if pd.api.types.is_numeric_dtype(cat_dtype) else '-999'

            if sentinel not in X_train[col].cat.categories:
                X_train[col] = X_train[col].cat.add_categories([sentinel])
            X_train[col] = X_train[col].fillna(sentinel)

            if not isinstance(X_test[col].dtype, pd.CategoricalDtype):
                X_test[col] = X_test[col].astype('category')
            if sentinel not in X_test[col].cat.categories:
                X_test[col] = X_test[col].cat.add_categories([sentinel])
            X_test[col] = X_test[col].fillna(sentinel)

        X_train[num_cols] = X_train[num_cols].fillna(-999)
        X_test[num_cols] = X_test[num_cols].fillna(-999)
        y_train = y_train.fillna(-999)
        y_test = y_test.fillna(-999)

elif missing_value_strategy == "impute":
    print("Imputing missing values with sentinel -999...")
    # Identify categorical vs numeric columns from train split.
    # For categoricals we add a sentinel category first, then fill missing.
    cat_cols = X_train.select_dtypes(include='category').columns
    num_cols = X_train.select_dtypes(exclude='category').columns

    # Fill categoricals: add sentinel category if needed, then fill.
    for col in cat_cols:
        cat_dtype = X_train[col].cat.categories.dtype
        sentinel = -999 if pd.api.types.is_numeric_dtype(cat_dtype) else '-999'

        if sentinel not in X_train[col].cat.categories:
            X_train[col] = X_train[col].cat.add_categories([sentinel])
        X_train[col] = X_train[col].fillna(sentinel)

        if not isinstance(X_test[col].dtype, pd.CategoricalDtype):
            X_test[col] = X_test[col].astype('category')
        if sentinel not in X_test[col].cat.categories:
            X_test[col] = X_test[col].cat.add_categories([sentinel])
        X_test[col] = X_test[col].fillna(sentinel)

    # Fill numerics and target with sentinel.
    X_train[num_cols] = X_train[num_cols].fillna(-999)
    X_test[num_cols] = X_test[num_cols].fillna(-999)
    y_train = y_train.fillna(-999)
    y_test = y_test.fillna(-999)

# Load best params
best_params = joblib.load(params_path / f'{run_id}_{model_name}{t}.pkl')

# Set pipeline to use best params
pipe.set_params(**best_params)

# Fit best pipeline to training data
print("Fitting model...")
pipe.fit(X_train, y_train)

# Evaluate on test data
y_pred = pipe.predict(X_test)
pipe_mae = np.mean(np.abs(y_test - y_pred))
pipe_mse = mean_squared_error(y_test, y_pred)
pipe_rmse = np.sqrt(pipe_mse)
pipe_r2 = r2_score(y_test, y_pred)

print(f"MAE: {pipe_mae:.3f}")
print(f"MSE: {pipe_mse:.3f}")
print(f"RMSE: {pipe_rmse:.3f}")
print(f"R²: {pipe_r2:.3f}")

# Create a sampler
print("Creating sampler...")
index_dict = {index:name for index, name in enumerate(X.columns)}
# Save mapping to make feature references explicit/reproducible across runs.
save_path.mkdir(parents=True, exist_ok=True)
with open(save_path / "index_dict.json", "w") as f:
    json.dump(index_dict, f, indent=4)

# Missing values are already handled above using `missing_value_strategy`.
# `handle_missing` below only controls sampler class family used by CIT.
# - False => RandomForest samplers
# - True  => HistGradientBoosting samplers
print(f"Missing-value strategy in use: {missing_value_strategy}")

if run_CIT == True:
    # Keep X as DataFrame (feature-name-based removal is used in CIT).
    # y is converted to ndarray since CIT inference consumes array-like targets.
    feature_names = list(X_train.columns) 

    # WARNING: this means that sklearn loses info on categorical features 
    # X_train = X_train.to_numpy()
    # X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()
    y_train = y_train.to_numpy()

    if sample_features == True:
        random.seed(seed)
        print(f"Sampling {n_features} features...")

        # Sample n random keys
        sample_keys = random.sample(list(index_dict.keys()), n_features)
        # Create a new dictionary with those keys
        index_dict = {k: index_dict[k] for k in sample_keys}

    # loop through all variables:
    counter = 1
    results = []
    print(f"Running {test} loop...")
    for removal, name in index_dict.items():

        if handle_missing == False:
            if index_dict[removal] in categorical_features_in_data:
                sampler = RandomForestClassifier()
            else:        
                sampler = RandomForestRegressor()

        if handle_missing == True:
            if index_dict[removal] in categorical_features_in_data:
                sampler = HistGradientBoostingClassifier()
            else:        
                sampler = HistGradientBoostingRegressor()

        if sample_features == True:
            print(f"{counter}/{len(index_dict)}: " + str(name))
        else:
            print(str(removal+1)+"/"+str(len(index_dict)) + ": " + str(name))
        
        # fit the sampler on all variables except the one to be removed
        # Sampler learns P(feature_j | X_-j) to generate conditional replacements.
        _ = sampler.fit(
            X_train.drop(name, axis=1),
            # np.delete(X_train, name, axis = 1), 
            # X_train[:, removal] --> if numpy arrays
            X_train[[name]]) # if DataFrame                        
        
        if test == "RPT":

            rpt = CIT( 
                learner = pipe, #  does take a pipe
                sampler = sampler,
                removal = removal,
                method = test,
                null_dist = "resampling", #  construct a null distribution by using resampling technique
                n_copies = n_iter, # number of null copies (conditional resamples) to create to make the null distribution
                random_state= seed,
                loss_func="mean_squared_error")

            _ = rpt.infer(X_test, y_test)
            result = rpt.summarize(cross_fit=None)

        if test == "CPI":
            cpi = CIT(
                learner = pipe, 
                sampler = sampler,
                removal = index_dict[removal], # for DataFrame need to use the feature name
                method = test,
                # "normality" is faster for large loops; "permutation" is heavier but more robust.
                null_dist="normality", # other option "permutation" -> then set n_permutations=__
                n_copies= 1,
                random_state = seed,
                loss_func="mean_squared_error")
            
            _ = cpi.infer(X_test, y_test)
            result = cpi.summarize(cross_fit=None)

        result = result.reset_index()   
        result["feature"] = name  #<- if using DataFrame           
        # result["feature"] = result["removal"].map(index_dict)  #<- if using numpy arrays

        # as get NA when SE is 0, fill as 1 for now:
        result.loc[result["std_error"] == 0, "p_value"] = result.loc[result["std_error"] == 0, "p_value"].fillna(1)
        # result["p_value"] = result["p_value"].fillna(1)

        if sample_features == True:
            counter +=1

        results.append(result)
        # Keep console output compact during long runs.
        print(f"Completed feature: {name}")

    # Add variable names back in
    result_df = pd.concat(results, ignore_index=True) 

    # Save results dataframe
    if sample_features == True:
        result_df.to_csv(save_path / f"{test}/{start_string}_{model_name}_{n_features}_features_results.csv")
    else:
        result_df.to_csv(save_path / f"{test}/{start_string}_{model_name}_results.csv")

# todo:
# "n categories in columns [1, 2, 3, ... 96] during transform.
# These unknown categories will be encoded as all zeros
# I expect get this due to the NAs -> need to handle better for categoricals
# **Note** this does not control for family wise error rate!!!!

# run diagnostics
if run_diagnostics == True:
    print("Running diagnostics...")
    # Use the same missing-value strategy for diagnostics input
    # so degeneracy checks reflect actual modeling assumptions.
    X_diag = X.copy()
    if missing_value_strategy == "drop":
        X_diag = X_diag.dropna(axis=0).copy()
    else:
        cat_cols = X_diag.select_dtypes(include='category').columns
        for col in cat_cols:
            cat_dtype = X_diag[col].cat.categories.dtype
            sentinel = -999 if pd.api.types.is_numeric_dtype(cat_dtype) else '-999'

            if sentinel not in X_diag[col].cat.categories:
                X_diag[col] = X_diag[col].cat.add_categories([sentinel])
            X_diag[col] = X_diag[col].fillna(sentinel)

            if not isinstance(X_diag[col].dtype, pd.CategoricalDtype):
                X_diag[col] = X_diag[col].astype('category')
            if sentinel not in X_diag[col].cat.categories:
                X_diag[col] = X_diag[col].cat.add_categories([sentinel])
            X_diag[col] = X_diag[col].fillna(sentinel)

        X_diag[numerical_features] = X_diag[numerical_features].fillna(-999)

    print("Running diagnostics on 0 SE...")
    if run_CIT == False:
        if sample_features == True:
            result_df = pd.read_csv(save_path / f"{test}/{start_string}_{model_name}_{n_features}_features_results.csv")
        else:
            result_df = pd.read_csv(save_path / f"{test}/{start_string}_{model_name}_results.csv")
    deg = diagnose_degenerate_features(result_df, X_diag)

    if sample_features == True:
        deg.to_csv(f"Sim-CIT/{test}/Diagnostics/" + f'{start_string}_{model_name}_{n_features}_degenerate_features.csv',
                index=False)
    else:   
        deg.to_csv(f"Sim-CIT/{test}/Diagnostics/" + f'{start_string}_{model_name}_all_degenerate_features.csv',
            index=False)

finish_time = dt.now()
print("Time taken:", finish_time - start_time)
print("Done")