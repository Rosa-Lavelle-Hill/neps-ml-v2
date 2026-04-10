# just use as a trial to see if it works
from datetime import datetime as dt
import json
import joblib
import random
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from Functions.CIT import diagnose_degenerate_features
from Functions.pipeline import construct_pipelines_all_no_imputation
from Functions.preprocessing_functions import drop_cols, remove_variables
from fixed_params import categorical_features, var_info_sheet, seed, test_size, remove_vars
from Params.Grids import dt_param_grid, rf_param_grid, hgb_param_grid, xgb_param_grid
from tofi import CIT

# set to true if want to reduce computation time to test code -- will only run for a sample of features
test = "CPI" # "RPT" or "CPI"
run_CIT = True 
sample_features = False # set to true if want to reduce computation time to test code -- will only run for a sample of features
run_diagnostics = True
n_iter = 1000
# start_string = '26_Nov_2024__16.45'
start_string = '12_Dec_2024__11.47_test'
test_run = True
model_name = "HGB" # model_names = ["DT", "RF", "HGB", "XGB"]
handle_missing = False # if True will use HGB as a sampler (would still need to have remove as -999, so leave as False for now); otherwise if False will use default RF and simply fill missing data with -999

if sample_features == True:
    n_features = 10 # number of features to sample
if test_run == True:
    t = '_test'
else:
    t = ''

start_time = dt.now()
save_path = "Sim-CIT/"
params_path = "Results/Prediction/Best_Params/"

# Import variable information & meta data:
var_info = pd.read_csv(var_info_sheet, encoding="utf-8", sep=';')
var_info = var_info[var_info["include as predictor"] == 1]
var_names_dict = dict(zip(var_info['var'], var_info['varname']))

# get block information for grouped importance
block_dict = dict(zip(var_info['var'], var_info['varsection']))

print("Loading data...")
X_and_y = pd.read_csv("Data/Preprocessed/X_and_y.csv", index_col=[0])

X = X_and_y.drop("y", axis=1)
print("X shape: " + str(X.shape))
y = X_and_y["y"]
print("y shape: " + str(y.shape))

# Ensure categorical features actually exist in X
categorical_features_list = categorical_features.values.flatten().tolist()
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
numerical_features_in_data = [elem for elem in numerical_features if elem in list(X.columns)]
print("Data contains {} numerical features".format(len(numerical_features_in_data)))
categorical_features_in_data = [elem for elem in categorical_features if elem in list(X.columns)]
print("Data contains {} categorical features".format(len(categorical_features_in_data)))

numeric_features_index = X[numerical_features_in_data].columns
categorical_features_index = X[categorical_features_in_data].columns

# Construct all pipelines with no imputation
print("Constructing pipelines...")
pipe_dt, pipe_rf, pipe_hgb, pipe_xgb = construct_pipelines_all_no_imputation(numeric_features_index, categorical_features_index)

if model_name == "DT":
    pipe = pipe_dt
elif model_name == "RF":
    pipe = pipe_rf
elif model_name == "HGB":
    pipe = pipe_hgb
elif model_name == "XGB":
    pipe = pipe_xgb

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=seed, test_size=test_size, shuffle=True)
print("X_train shape: " + str(X_train.shape))
print("X_test shape: " + str(X_test.shape))

# Load best params
best_params = joblib.load(params_path + f'{start_string}_{model_name}{t}.pkl')

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
with open(save_path+"index_dict.json", "w") as f:
    json.dump(index_dict, f, indent=4)

# sampler does not work on missing data so first need to impute data, or to keep missing distribute fill with constant
if handle_missing == False:
    print("Filling missing data with -999...")
    # Identify categorical vs numeric columns
    cat_cols = X_train.select_dtypes(include='category').columns
    num_cols = X_train.select_dtypes(exclude='category').columns

# Fill categoricals: add sentinel category if needed, then fill
for col in cat_cols:
    cat_dtype = X_train[col].cat.categories.dtype
    sentinel = -999 if pd.api.types.is_numeric_dtype(cat_dtype) else '-999'

    # Train
    if sentinel not in X_train[col].cat.categories:
        X_train[col] = X_train[col].cat.add_categories([sentinel])
    X_train[col] = X_train[col].fillna(sentinel)

    # Test: ensure categorical dtype first, then add+fill
    if not pd.api.types.is_categorical_dtype(X_test[col]):
        X_test[col] = X_test[col].astype('category')
    if sentinel not in X_test[col].cat.categories:
        X_test[col] = X_test[col].cat.add_categories([sentinel])
    X_test[col] = X_test[col].fillna(sentinel)

# Fill numerics
X_train[num_cols] = X_train[num_cols].fillna(-999)
X_test[num_cols]  = X_test[num_cols].fillna(-999)

# Target
y_test = y_test.fillna(-999)

if run_CIT == True:
    # convert to numpy arrays for CIT
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
                null_dist="normality", #  construct a null distribution by assuming normality, other option "permutation" -> then set n_permutations=__
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
        print(results)

    # Add variable names back in
    result_df = pd.concat(results, ignore_index=True) 

    # Save results dataframe
    if sample_features == True:
        result_df.to_csv(save_path + f"{test}/{start_string}_{model_name}_{n_features}_features_results.csv")
    else:
        result_df.to_csv(save_path + f"{test}/{start_string}_{model_name}_results.csv")

# todo:
# "n categories in columns [1, 2, 3, ... 96] during transform.
# These unknown categories will be encoded as all zeros
# I expect get this due to the NAs -> need to handle better for categoricals
# **Note** this does not control for family wise error rate!!!!

# run diagnostics
if run_diagnostics == True:
    print("Running diagnostics...")
    # X_imp =  X.fillna(-999)

    for col in cat_cols:
        cat_dtype = X[col].cat.categories.dtype
        sentinel = -999 if pd.api.types.is_numeric_dtype(cat_dtype) else '-999'

        # Train
        if sentinel not in X[col].cat.categories:
            X[col] = X[col].cat.add_categories([sentinel])
        X[col] = X[col].fillna(sentinel)

        # Test: ensure categorical dtype first, then add+fill
        if not pd.api.types.is_categorical_dtype(X[col]):
            X[col] = X[col].astype('category')
        if sentinel not in X[col].cat.categories:
            X[col] = X[col].cat.add_categories([sentinel])
        X[col] = X[col].fillna(sentinel)
    X[numerical_features] = X[numerical_features].fillna(-999)

    print("Running diagnostics on 0 SE...")
    if run_CIT == False:
        result_df = pd.read_csv(save_path + f"{test}/{start_string}_{model_name}_{n_features}_features_results.csv")
    deg = diagnose_degenerate_features(result_df, X)

    if sample_features == True:
        deg.to_csv(f"Sim-CIT/{test}/Diagnostics/" + f'{start_string}_{model_name}_{n_features}_degenerate_features.csv',
                index=False)
    else:   
        deg.to_csv(f"Sim-CIT/{test}/Diagnostics/" + f'{start_string}_{model_name}_all_degenerate_features.csv',
            index=False)

finish_time = dt.now()
print("Time taken:", finish_time - start_time)
print("Done")