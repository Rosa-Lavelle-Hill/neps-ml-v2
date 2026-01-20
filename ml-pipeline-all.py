# Use only models that handle missing data as inputs
import pickle
import joblib
import json
import re
import datetime as dt
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

from Functions.grouped_importance import group_permutation_analysis, group_permutation_analysis_avg
from Functions.plotting import plot_label_reg_sns, plot_scat, plot_permutation, plot_SHAP, plot_permutation_bars, \
    plot_results, plot_group_perm_importance, plot_group_SHAP_importance
from Functions.pipeline import get_preprocessed_col_names, construct_pipelines_no_imputation, construct_pipelines, \
    construct_pipelines_all_no_imputation, get_preprocessed_data
from Functions.preprocessing_functions import drop_cols, count_categories_to_file, remove_variables

# import fixed params:
from fixed_params import (categorical_features, seed, n_permutations, test_n_permutations, test_size, cv, scoring,
                          decimal_places, plot_n_features, smallest_category_count, remove_vars, test_imputer_max_iter,
                          test_cv, test_n_permutations, dv_t1_name, var_info_sheet, school_track)

categorical_features = list(categorical_features['0'])

# Global Boolean run params:
TEST_RUN = True # runs test run of code (fixed grid)
count_categories = False # counts min categories and notes where <25
train_models = True # trains models, if False uses optimal model parameters saved from a previous run (use "start_string" to choose previous run)
test_models = True # runs evaluation of model performance
interpret_models = True # runs model interpretation, if False then neither SHAP nor permutation importance will run
run_permutation = True # runs permutation importance, otherwise loads results and plots
run_grouped_permutation = True # calculates and plots grouped permutation importance
run_SHAP = True # runs SHAP, otherwise loads results and plots
run_grouped_SHAP = True # calculates and plots grouped SHAP
plot_nice_names = True

# todo: UserWarning: Found unknown categories in columns [23] during transform. These unknown categories will be encoded as all zeros -- I think do do with NAs -- what happens in this case?
# todo: RA - newly created variables and others need a "nice plot names"
# todo: clean up grids.py, and try come up up with final reasonable suggestions

# save paths:
results_path = "Results/"
params_save = results_path + "Prediction/Best_Params/"
plot_save = results_path + "Prediction/Plots/"
outputs_path = "Outputs/"

# Read data
X_and_y = pd.read_csv("Data/Preprocessed/X_and_y.csv", index_col=[0])

X = X_and_y.drop("y", axis=1)
y = X_and_y["y"]

if TEST_RUN == True:
    print("running a test run.... df shape is " + str(X_and_y.shape))
    from Params.Grids import test_dt_param_grid, test_rf_param_grid, test_hgb_param_grid, test_xgb_param_grid
    param_list = [test_dt_param_grid, test_rf_param_grid, test_hgb_param_grid, test_xgb_param_grid]
    run = "_test"
    n_permutations = test_n_permutations
    cv = test_cv
    imputer_max_iter = test_imputer_max_iter
else:
    from Params.Grids import dt_param_grid, rf_param_grid, hgb_param_grid, xgb_param_grid
    param_list = [dt_param_grid, rf_param_grid, hgb_param_grid, xgb_param_grid]
    run = ""

# start signature
script_start = dt.datetime.now()
if train_models == True:
    start_string = script_start.strftime('%d_%b_%Y__%H.%M{}'.format(run))
else:
    start_string = '26_Nov_2024__16.45' # <-- declare start string here depending on what model run want to test and evaluate

# Assign data types
numerical_df, numerical_features = drop_cols(categorical_features, X)

categorical_features_list = categorical_features.values.flatten().tolist()
categorical_features = [c for c in categorical_features_list if c in X.columns]


# Redefine data types
X[categorical_features] = X[categorical_features].astype('category')
X[numerical_features] = X[numerical_features].astype('float')

# check all drop vars are removed
X = remove_variables(X, remove_vars)

# Check category counts across train/test data
numerical_features_in_data = [elem for elem in numerical_features if elem in list(X.columns)]
categorical_features_in_data = [elem for elem in categorical_features if elem in list(X.columns)]

numeric_features_index = X[numerical_features_in_data].columns
categorical_features_index = X[categorical_features_in_data].columns

# Construct all pipelines with no imputation
pipe_dt, pipe_rf, pipe_hgb, pipe_xgb = construct_pipelines_all_no_imputation(numeric_features_index, categorical_features_index)

# ******* define models ********
pipes = [pipe_dt, pipe_rf, pipe_hgb, pipe_xgb]
model_names = ["DT", "RF", "HGB", "XGB"]
# pipes = [pipe_dt]
# model_names = ["DT"]
# param_list = [dt_param_grid]

# Import variable information & meta data:
var_info = pd.read_csv(var_info_sheet, encoding="utf-8", sep=';')
var_info = var_info[var_info["include as predictor"] == 1]
var_names_dict = dict(zip(var_info['var'], var_info['varname']))

# get block information for grouped importance
block_dict = dict(zip(var_info['var'], var_info['varsection']))

# Create category_number:category_name dict
with open('Data/Meta/label_info_nested_dict.json', 'r') as file:
    cat_name_dict = json.load(file)

#=======================================================================================================================
# ML Pipeline...

# define empty dicts to save values in
best_params_dict = {}
test_scores = {}

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size, shuffle=True)

# Count category distributions across train and test data sets
save_path = outputs_path + "category_distributions/"
if count_categories == True:
    count_categories_to_file(X_train, output_file=save_path+"X_train_counts.txt", categorical_columns=categorical_features_in_data,
                             var_name_dict=var_names_dict, min_cat=round((smallest_category_count/100)*80, 0),
                             cat_name_dict=cat_name_dict, data_name="X_train")
    count_categories_to_file(X_test, output_file=save_path+"X_test_counts.txt", categorical_columns=categorical_features_in_data,
                             var_name_dict=var_names_dict, min_cat=round((smallest_category_count/100)*20, 0),
                             cat_name_dict=cat_name_dict, data_name="X_test")

    # plot cardinality of features
    cardinality = X[categorical_features_in_data].nunique()
    cardinality = cardinality[cardinality > 2]
    cardinality = cardinality.sort_values(ascending=False)
    plt.figure(figsize=(6, 6))
    cardinality.plot(kind='bar')
    plt.title('Cardinality of Categorical Features > 2')
    plt.xlabel('Features')
    plt.ylabel('Number of Unique Values')
    plt.tight_layout()
    plt.savefig(outputs_path + "category_distributions/cardinality.png")


# loop through one model class at a time
for model_name, pipe, params in zip(model_names, pipes, param_list):
    model_train_start = dt.datetime.now()

    save_file = results_path + f"/Prediction/{start_string}_{model_name}{run}.txt"

    if train_models == True:
        print("{}: ".format(model_name), file=open(save_file, "w"))
        print("Running {} model".format(model_name))

        # Perform CV on train data to tune model hyper-parameters
        grid_search = GridSearchCV(estimator=pipe,
                                   param_grid=params,
                                   cv=cv,
                                   scoring=scoring,
                                   refit=False,
                                   verbose=2,
                                   n_jobs=2,
                                   error_score='raise')

        # start timer
        grid_start = dt.datetime.now()
        # run the grid search
        grid_search.fit(X_train, y_train)
        # end timer
        grid_end = dt.datetime.now()
        training_time = grid_end - grid_start
        print("Training done. Time taken: {}".format(training_time), file=open(save_file, "a"))

        # Store best hyper-paramters:
        best_params = grid_search.best_params_
        joblib.dump(best_params, params_save + f'{start_string}_{model_name}{run}.pkl', compress=1)
        best_train_score = round(abs(grid_search.best_score_), decimal_places)
        print("params tried:\n{}\n".format(params), file=open(save_file, "a"))

        print("Best training {} score: {}. Best model params:\n{}.\n".format(scoring, best_train_score, best_params),
            file=open(save_file, "a"))

        best_params_dict[model_name] = best_params
    else:
        best_params = joblib.load(params_save + f'{start_string}_{model_name}.pkl')

    # set pipeline to use best params
    pipe.set_params(**best_params)

    # fit best pipeline to training data
    pipe.fit(X_train, y_train)

    model_train_end = dt.datetime.now()
    model_train_time = model_train_end - model_train_start
    print(f"Model train time for {model_name}: {model_train_time}")
    #========================================
    # Test best model on hold-out test data
    if test_models == True:
        model_test_start = dt.datetime.now()

        # Add a baseline model 1: dv at T1 (linear regression) -----------------------------------------

        lr = LinearRegression()
        lr.fit(pd.DataFrame(X_train[dv_t1_name]), y_train)

        y_pred = lr.predict(pd.DataFrame(X_test[dv_t1_name]))

        # Calculate evaluation metrics
        dvt1_r_squared = round(r2_score(y_test, y_pred), 2)  # R-squared
        dvt1_rmse = round(root_mean_squared_error(y_test, y_pred), 2)  # Root Mean Squared Error
        dvt1_mae = round(mean_absolute_error(y_test, y_pred), 2)  # Mean Absolute Error

        # Add a baseline model 2: dv at T1 and track (linear regression) -------------------------------

        lr2 = LinearRegression()
        lr2.fit(pd.DataFrame(X_train[[dv_t1_name, school_track]]), y_train)

        y_pred2 = lr2.predict(pd.DataFrame(X_test[[dv_t1_name, school_track]]))

        # Calculate evaluation metrics
        dvt1_r_squared2 = round(r2_score(y_test, y_pred2), 2)  # R-squared
        dvt1_rmse2 = round(root_mean_squared_error(y_test, y_pred2), 2)  # Root Mean Squared Error
        dvt1_mae2 = round(mean_absolute_error(y_test, y_pred2), 2)  # Mean Absolute Error

        # ===================================================================================================
        test_scores["Prior Achieve. Baseline"] = {"R2": dvt1_r_squared, "MAE": dvt1_mae, "RMSE": dvt1_rmse}
        test_scores["Prior Ach.+ School Track"] = {"R2": dvt1_r_squared2, "MAE": dvt1_mae2, "RMSE": dvt1_rmse2}

        # Test model on hold-out data:
        print("Evaluating performance on test set for {}".format(model_name))

        # use pipeline to make predictions
        y_pred = pipe.predict(X_test)

        # evaluate/score best out of sample
        test_score_r2 = round(metrics.r2_score(y_test, y_pred), decimal_places)
        test_score_mae = round(metrics.mean_absolute_error(y_test, y_pred), decimal_places)
        test_score_rmse = round(metrics.root_mean_squared_error(y_test, y_pred), decimal_places)

        print(f"Best {model_name} model performance on test data:\nR2: {test_score_r2}; mae: {test_score_mae}",
              file=open(save_file, "a"))

        test_scores[model_name] = {"R2": test_score_r2, "MAE": test_score_mae, "RMSE": dvt1_rmse}

        # plot distribution of predictions:
        plot_scat(x=y_test, y=y_pred, x_lab="actual", y_lab="predicted",
                  save_path=plot_save,
                  save_name=f"{start_string}_{model_name}_predicted_actual{run}")

        plot_label_reg_sns(x=y_test, y=y_pred, x_lab="actual", y_lab="predicted",
                           save_path=plot_save,
                           save_name=f"{start_string}_{model_name}_predicted_actual_cor{run}", anov_var=None,
                           cor=True, ano=False, print_cor=False, same_axis=True)

        model_test_end = dt.datetime.now()
        model_test_time = model_test_end - model_test_start
        print(f"Model test time for {model_name}: {model_test_time}")

    # ===========================================================================================
    # Interpretations of best model on test data

    if interpret_models == True:

        model_interpretation_start = dt.datetime.now()
        save_path = results_path + "Interpretation/Permutation/"

        # Fit the preprocessor
        opt_model = pipe.named_steps.regressor
        preprocessor = pipe.named_steps.preprocessor

        # fit to and transform train
        X_train_p, X_test_p = get_preprocessed_data(
            pipeline=pipe_dt,
            X_train=X_train,
            X_test=X_test,
            numeric_features=numerical_features_in_data,
            categorical_features=categorical_features_in_data)

        names = X_test_p.columns

        if plot_nice_names == True:
            processed_names = list(X_test_p.columns)

            # Initialize the result dictionary
            processed_rename_dict = {}

            # Regular expression to match the structure "prefix_number"
            pattern = re.compile(r"^(.*?)(_([0-9.]+))?$")

            # Process each name in the list
            for name in processed_names:
                match = pattern.match(name)
                if match:
                    prefix = match.group(1)  # Everything before the number
                    suffix = match.group(3)  # The number part, if present

                    # Check if the prefix is in the rename dictionary
                    if prefix in var_names_dict:
                        # Get the new prefix and strip any trailing whitespace
                        new_prefix = str(var_names_dict[prefix]).rstrip()
                        # Combine new prefix with suffix, separated by a space
                        new_name = f"{new_prefix} {suffix}" if suffix else new_prefix
                        processed_rename_dict[name] = new_name

        if run_permutation == True:

            # 1) Permutation importance
            print("Starting permutation importance for {}".format(model_name))
            result = permutation_importance(pipe, X_test, y_test, n_repeats=n_permutations,
                                            random_state=seed, n_jobs=2, scoring=scoring)
            perm_importances_mean = result.importances_mean
            vars=list(X_test.columns)
            perm_imp_df = pd.DataFrame({
                'Feature': vars,
                'importance_mean': result['importances_mean'],
                'importance_std': result['importances_std']
            })

            # take top n features and then flip so most important at top on graph
            perm_imp_df.sort_values(by="importance_mean", ascending=False, inplace=True, axis=0)
            perm_imp_df.to_csv(save_path+f"{start_string}_{model_name}_permutation_importance{run}.csv")

        else:
            perm_imp_df = pd.read_csv(save_path + f"{start_string}_{model_name}_permutation_importance{run}.csv",
                                      index_col=[0])
            if plot_nice_names == True:
                perm_imp_df["Original Feature Name"] = perm_imp_df["Feature"].copy()
                perm_imp_df["Feature"].replace(var_names_dict, inplace=True)

        # plot
        vars = list(X_test.columns)
        perm_imp_df = perm_imp_df[0:plot_n_features]
        perm_imp_df.sort_values(by="importance_mean", ascending=True, inplace=True, axis=0)

        plot_permutation(perm_imp_df=perm_imp_df,
                         save_path=save_path + "PLots/",
                         save_name=f"{start_string}_{model_name}_permutation{run}")

        if n_permutations > 1:
            plot_permutation_bars(perm_imp_df=perm_imp_df, save_path=save_path + "PLots/",
                                  save_name=f"{start_string}_{model_name}_MULTIpermutation{run}",
                                  plot_n_features=plot_n_features)

        if run_grouped_permutation == True:
            print("Starting grouped permutation importance")

            save_path = results_path + "Interpretation/Permutation/Grouped/"
            result = group_permutation_analysis_avg(X_train, y_train,
                                                X_test, y_test,
                                                pipeline=pipe,
                                                save_path=save_path,
                                                model_name=model_name,
                                                group_dict=block_dict,
                                                n=n_permutations)

            result_df = pd.DataFrame.from_dict(result, orient='index', columns=["Importance"])
            result_df.sort_values(by="Importance", ascending=False, inplace=True, axis=0)
            result_df.to_csv(save_path + f"Grouped_{start_string}_{model_name}{run}.csv")

            plot_group_perm_importance(result, save_path=save_path + "PLots/",
                                       save_name=f"Grouped_{start_string}_{model_name}{run}"
                                       )

        # 2) SHAP importance ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        save_path= results_path + "Interpretation/SHAP/"
        if run_SHAP == True:

            print("Starting SHAP importance for {}".format(model_name))

            # fit optimised model to transformed train data
            opt_model.fit(X_train_p, y_train)

            # Fit the explainer
            shap_results_dict = {}

            # method_types = ["tree_path_dependent", "interventional"]# (see Lundberg et al., 2020) https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
            # data_options = [None, shap.sample(X_test_p, 1000)]
            method_types = ["interventional"]
            data_options = [shap.sample(X_test_p, 1000)]
            for method_type, data_option in zip(method_types, data_options):
                print(method_type)
                if (model_name == 'DT'):
                    explainer = shap.TreeExplainer(model=opt_model, data=data_option, feature_perturbation=method_type)

                else:
                    explainer = shap.KernelExplainer(
                        model=lambda x: opt_model.predict(x),
                        data=shap.sample(X_test_p, 100)
                    )

                # Calculate the SHAP values and save
                shap_dict = explainer(X_test_p)
                shap_values = explainer.shap_values(X_test_p)

                # save
                shap_values_df = pd.DataFrame(shap_values, columns=names)
                shap_values_df.to_csv(save_path + f"{start_string}_SHAP_{model_name}-{method_type}{run}.csv")
                shap_results_dict[method_type] = shap_dict

                file_path = save_path + f"{start_string}_SHAP_{model_name}{run}.pkl"
                with open(file_path, 'wb') as handle:
                    pickle.dump(shap_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            method_type = "interventional" # todo: check if we want to include "tree path dependent"
            shap_values_df = pd.read_csv(save_path + f"{start_string}_SHAP_{model_name}-{method_type}{run}.csv")
            file_path = save_path + f"{start_string}_SHAP_{model_name}{run}.pkl"
            with open(file_path, 'rb') as handle:
                shap_results_dict = pickle.load(handle)

        # Plot
        if plot_nice_names == True:
            X_test_p_original_names = X_test_p.copy()
            X_test_p.rename(columns=processed_rename_dict, inplace=True)
            names = list(X_test_p.columns)

        shap_plot_save_path = save_path + "Plots/"
        for method, shap_dict in shap_results_dict.items():
            plot_types = ["bar", "summary", "violin"]
            for plot_type in plot_types:
                print(plot_type)
                plot_SHAP(shap_dict, col_list=names, data=X_test_p,
                          n_features=plot_n_features, plot_type=plot_type,
                          save_path=shap_plot_save_path, title="SHAP importance (test set)",
                          save_name=f"{start_string}_{model_name}_SHAP_{plot_type}_{method}{run}.png")
                if plot_type == "summary":
                    plot_SHAP(shap_dict, col_list=names, data=X_test_p,
                              n_features=plot_n_features, plot_type=None,
                              save_path=shap_plot_save_path, title="SHAP importance (test set)",
                              save_name=f"{start_string}_{model_name}_SHAP_{plot_type}_{method}{run}.png")

        if run_grouped_SHAP == True:
            print('running grouped SHAP importance...')
            # Get extended mapping dict

            # Regex to remove '_{any digit}.0' or '_nan' at the end of the column name
            base_columns = [re.sub(r'(_\d+\.0|_nan)$', '', col) for col in X_train_p.columns]

            # Create a mapping dictionary for the extended column names
            extended_mapping = {col: block_dict.get(base_col, 'Unmapped') for col, base_col in
                                zip(X_train_p.columns, base_columns)}

            with open('Data/Meta/extended_group_mapping.json', 'w') as f:
                json.dump(extended_mapping, f)

            # Map each column to its group
            shap_values_df.columns = shap_values_df.columns.map(extended_mapping)
            # todo: update var_info some are unmapped^^^

            # Calculate the absolute SHAP values and sum them by group
            group_shap_values = (
                shap_values_df.abs()  # Take absolute values
                .sum(axis=0)  # Sum across rows for each column
                .groupby(level=0)  # Group by the mapped group names
                .sum()  # Aggregate within each group
                .round(2)
            )
            method_type = "interventional"
            # Save values
            group_shap_values_df = pd.DataFrame(group_shap_values, columns=["Importance"])
            group_shap_values_df.sort_values(by="Importance", ascending=False, inplace=True, axis=0)
            group_shap_values_df.to_csv("Results/Interpretation/SHAP/Grouped/" +
                                     f"Grouped_{start_string}_{model_name}_{method_type}{run}.csv")
            # Plot
            plot_group_SHAP_importance(group_shap_values, f"Grouped_{start_string}_{model_name}_{method_type}{run}",
                                       save_path= "Results/Interpretation/SHAP/Grouped/Plots/")

        model_interpretation_end = dt.datetime.now()
        model_interpretation_time = model_interpretation_end - model_interpretation_start
        print(f"Model interpretation time for {model_name}: {model_interpretation_time}")

if test_models == True:
    results_df = pd.DataFrame.from_dict(test_scores)
    results_df.to_csv(f"Results/Prediction/All_models/all_test_scores_{start_string}.csv")

if test_models == False:
    results_df = pd.read_csv(f"Results/Prediction/All_Models/all_test_scores_{start_string}{run}.csv", index_col=[0])
    print(f"Loading test results data from: Results/Prediction/All_models/all_test_scores_{start_string}{run}.csv")

print("Plotting test performance to compare all models...")

print(f"Loading test results data from: Results/Prediction/All_Models/all_test_scores_{start_string}{run}.csv")
x_ticks = ["Prior Achieve.", "Prior Ach.+ School Track" "Decision Tree", "Random Forest", "Hist Grad. Boost.", "XGBoost"]

plot_results(y="R2", data=results_df, colour='Model',
             save_path="Results/Prediction/All_Models/Plots/",
             save_name=f"test_summary_all_R2_{start_string}{run}",
             xlab="Prediction Models", ylab="Prediction R Squared",
             title="",
             x_ticks=x_ticks)
plot_results(y="MAE", data=results_df, colour='Model',
             save_path="Results/Prediction/All_Models/Plots/",
             save_name=f"test_summary_all_MAE_{start_string}{run}",
             xlab="Prediction Models", ylab="MAE",
             title="",
             x_ticks=x_ticks)
plot_results(y="RMSE", data=results_df, colour='Model',
             save_path="Results/Prediction/All_Models/Plots/",
             save_name=f"test_summary_all_RMSE_{start_string}{run}",
             xlab="Prediction Models", ylab="RMSE",
             title="",
             x_ticks=x_ticks)

# todo: add plot comparison plot of all models

# end timer
script_end = dt.datetime.now()
run_time = script_end - script_start
print(f"Script run time: {run_time}")
print("done!")