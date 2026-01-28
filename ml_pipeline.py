
def run_pipeline(cfg: dict):

    from pathlib import Path        
    import pickle
    import joblib
    import json
    import re
    import datetime as dt
    import pandas as pd
    import shap
    import matplotlib.pyplot as plt

    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

    from Functions.subsets import apply_subset
    from Functions.subsets import cfg_get, get_run_dir
    from Functions.grouped_importance import group_permutation_analysis_avg
    from Functions.plotting import plot_label_reg_sns, plot_scat, plot_permutation, plot_SHAP, plot_permutation_bars, \
                                    plot_results, plot_group_perm_importance, plot_group_SHAP_importance
    from Functions.pipeline import construct_pipelines_all_no_imputation, get_preprocessed_data
    from Functions.preprocessing_functions import drop_cols, count_categories_to_file, remove_variables

    from Params.Grids import get_test_param_grids, get_param_grids

    from fixed_params import (remove_vars, school_track, dv_t1_name)
    
    # =======================================================================================================
    script_start = dt.datetime.now()
    # =======================================================================================================
    # Extract run and load configs
    run_cfg = cfg.get("run", {})
    load_cfg = cfg.get("load", {})

    current_run_id = run_cfg["id"]  # where new outputs go
    if not current_run_id:
        raise RuntimeError("cfg['run']['id'] is missing")
    load_enabled = load_cfg.get("enabled", False) # where inputs can be loaded from

    if load_enabled:
        source_run_id = load_cfg.get("run_id")
        if not source_run_id:
            raise ValueError("load.enabled=True but load.run_id is not set")
    else:
        source_run_id = current_run_id
    # =======================================================================================================
    # import fixed params from base yaml:
    seed = cfg_get(cfg, ["params", "seed"])
    decimal_places = cfg_get(cfg, ["params", "decimal_places"])

    # run option flags:
    TEST_RUN = cfg_get(cfg, ["run", "test_run"])
    train_models = cfg_get(cfg, ["run", "train_models"])
    test_models = cfg_get(cfg, ["run", "test_models"])
    interpret_models = cfg_get(cfg, ["run", "interpret_models"])
    run_permutation = cfg_get(cfg, ["run", "run_permutation"])
    run_grouped_permutation = cfg_get(cfg, ["run", "run_grouped_permutation"])
    run_SHAP = cfg_get(cfg, ["run", "run_SHAP"])
    run_grouped_SHAP = cfg_get(cfg, ["run", "run_grouped_SHAP"])
    plot_nice_names = cfg_get(cfg, ["run", "plot_nice_names"])
    count_categories = cfg_get(cfg, ["run", "count_categories"])

    # paths:
    var_info_sheet = cfg_get(cfg, ["paths", "var_info_csv"])
    categorical_features_csv = cfg_get(cfg, ["paths", "categorical_features_csv"])
    preprocessed_data = cfg_get(cfg, ["paths", "preprocessed_data"])

    # preprocessing:
    smallest_category_count = cfg_get(cfg, ["preprocessing", "smallest_category_count"])

    # modelling:
    test_size = cfg_get(cfg, ["modelling", "test_size"])
    scoring = cfg_get(cfg, ["modelling", "scoring"])
    cv = cfg_get(cfg, ["modelling", "cv"])
    skl_n_jobs = cfg_get(cfg, ["modelling", "skl_n_jobs"])

    # interpretation:
    plot_n_features = cfg_get(cfg, ["interpretation", "plot_n_features"])
    n_permutations = cfg_get(cfg, ["interpretation", "n_permutations"])

    # test run params:
    test_cv = cfg_get(cfg, ["test_run_params", "test_cv"])
    test_n_permutations = cfg_get(cfg, ["test_run_params", "test_n_permutations"])

    # run-scoped save paths
    results_dir = get_run_dir(source_run_id, create=True)
    print("Saving to:", results_dir)

    results_path = Path(results_dir)
    params_save = results_path / "Prediction" / "Best_Params"
    plot_save = results_path / "Prediction" / "Plots"
    all_models_save = results_path / "Prediction" / "All_Models"
    shap_save = results_path / "Interpretation" / "SHAP"
    perm_imp_save = results_path / "Interpretation" / "Permutation"

    # ensure directories exist
    for p in [params_save, plot_save, all_models_save, shap_save, perm_imp_save]:
        p.mkdir(parents=True, exist_ok=True)

    # load results paths
    load_enabled = load_cfg.get("enabled", False)

    # determine source run id (what to load from)
    if load_enabled:
        source_run_id = load_cfg.get("run_id")
        if not source_run_id:
            raise ValueError("load.enabled=True but load.run_id is not set")
    else:
        source_run_id = current_run_id  # load from self by default

    # compute load dir (do NOT create)
    results_load_dir = get_run_dir(source_run_id, create=False)

    if load_enabled:
        if not results_load_dir.exists():
            raise FileNotFoundError(f"Requested load run does not exist: {results_load_dir}")
        print("Loading results from:", results_load_dir)

    perm_imp_load = results_load_dir / "Interpretation" / "Permutation" 
    shap_load = results_load_dir / "Interpretation" / "SHAP"
    all_models_load = results_load_dir / "Prediction" / "All_Models"

    # general outputs path
    outputs_path = Path("Outputs")

    # =======================================================================================================

    # Read data
    X_and_y = pd.read_csv(preprocessed_data, index_col=[0])
    X = X_and_y.drop("y", axis=1)
    y = X_and_y["y"]
    print("Initial X shape is: " + str(X.shape))

    # Import variable information & meta data:
    var_info = pd.read_csv(var_info_sheet, encoding="utf-8", sep=None, engine="python")
    var_info.columns = var_info.columns.str.replace("\ufeff", "", regex=False).str.strip()

    # var_info = pd.read_csv(var_info_sheet, encoding="utf-8", sep=';')
    var_info = var_info[var_info["include as predictor"] == 1]
    var_names_dict = dict(zip(var_info['var'], var_info['varname']))

    # get block information for grouped importance
    block_dict = dict(zip(var_info['var'], var_info['Variable_Group']))

    # Create category_number:category_name dict
    with open('Data/Meta/label_info_nested_dict.json', 'r') as file:
        cat_name_dict = json.load(file)

    # Define subset using yaml
    X = apply_subset(X, var_info, cfg)
    X_and_y = apply_subset(X_and_y, var_info, cfg)   # cfg is the loaded yaml config
    print("After applying subset, X shape is: " + str(X.shape))
    #======================================================================================================= 
    if TEST_RUN == True:
        print("running a test run.... df shape is " + str(X_and_y.shape))       
        param_list = get_test_param_grids(seed)
        run = "_test"
        n_permutations = test_n_permutations
        cv = test_cv
    else:
        param_list = get_param_grids(seed)
        run = ""

    # ========================================
    # Define run and load labels
    # used ONLY for saving filenames (optional)
    run_label = current_run_id

    # used ONLY for loading previous results
    load_label = source_run_id
    params_load = Path("Results") / "runs" / load_label / "Prediction" / "Best_Params"
    all_models_load = Path("Results") / "runs" / load_label / "Prediction" / "All_Models"
    # ========================================

    # Assign data types
    categorical_features = pd.read_csv(categorical_features_csv, index_col=[0]) # i
    categorical_features = list(categorical_features['0'])
    numerical_df, numerical_features = drop_cols(categorical_features, X)
    categorical_features = [c for c in categorical_features if c in X.columns]

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
    pipe_dt, pipe_rf, pipe_hgb, pipe_xgb = construct_pipelines_all_no_imputation(categorical_features_index)

    # ******* define models ********
    pipes = [pipe_dt, pipe_rf, pipe_hgb, pipe_xgb]
    model_names = ["DT", "RF", "HGB", "XGB"]
    # pipes = [pipe_dt]
    # model_names = ["DT"]
    # param_list = [dt_param_grid]

    #=======================================================================================================================
    # ML Pipeline...

    # define empty dicts to save values in
    best_params_dict = {}
    test_scores = {}

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size, shuffle=True)

    # Count category distributions across train and test data sets
    cat_save_path = outputs_path / "category_distributions"
    if count_categories == True:
        count_categories_to_file(X_train, output_file=cat_save_path+"X_train_counts.txt", categorical_columns=categorical_features_in_data,
                                var_name_dict=var_names_dict, min_cat=round((smallest_category_count/100)*80, 0),
                                cat_name_dict=cat_name_dict, data_name="X_train")
        count_categories_to_file(X_test, output_file=cat_save_path+"X_test_counts.txt", categorical_columns=categorical_features_in_data,
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

        save_file = results_path / f"Prediction/{run_label}_{model_name}{run}.txt"

        if train_models == True:
            print("{}: X_train = {}; X_test = {}".format(run_label, X_train.shape, X_test.shape), file=open(save_file, "w"))
            print("{}: ".format(model_name), file=open(save_file, "a"))
            print("Running {} model".format(model_name))

            # Perform CV on train data to tune model hyper-parameters
            grid_search = GridSearchCV(estimator=pipe,
                                    param_grid=params,
                                    cv=cv,
                                    scoring=scoring,
                                    refit=False,
                                    verbose=2,
                                    n_jobs=skl_n_jobs,
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
            joblib.dump(best_params, params_save / f'{run_label}_{model_name}{run}.pkl', compress=1)
            best_train_score = round(abs(grid_search.best_score_), decimal_places)
            print("params tried:\n{}\n".format(params), file=open(save_file, "a"))

            print("Best training {} score: {}. Best model params:\n{}.\n".format(scoring, best_train_score, best_params),
                file=open(save_file, "a"))

            best_params_dict[model_name] = best_params
        else:
            print("Loading best parameters for {} model from previous run...{}".format(model_name, load_label))
            filename = f'{load_label}_{model_name}{run}.pkl'
            best_params = joblib.load(params_load / filename )

        # set pipeline to use best params
        pipe.set_params(**best_params)

        # fit best pipeline to training data
        print("Fitting best {} model to full training data...".format(model_name))
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
            test_scores["Prior Ach. Baseline"] = {"R2": dvt1_r_squared, "MAE": dvt1_mae, "RMSE": dvt1_rmse}
            test_scores["Prior Ach.+ Track"] = {"R2": dvt1_r_squared2, "MAE": dvt1_mae2, "RMSE": dvt1_rmse2}

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

            test_scores[model_name] = {"R2": test_score_r2, "MAE": test_score_mae, "RMSE": test_score_rmse}

            # plot distribution of predictions:
            plot_scat(x=y_test, y=y_pred, x_lab="actual", y_lab="predicted",
                    save_path=plot_save,
                    save_name=f"{run_label}_{model_name}_predicted_actual{run}")

            plot_label_reg_sns(x=y_test, y=y_pred, x_lab="actual", y_lab="predicted",
                            save_path=plot_save,
                            save_name=f"{run_label}_{model_name}_predicted_actual_cor{run}", anov_var=None,
                            cor=True, ano=False, print_cor=False, same_axis=True)

            model_test_end = dt.datetime.now()
            model_test_time = model_test_end - model_test_start
            print(f"Model test time for {model_name}: {model_test_time}")
        else:
            print("Skipping model testing on hold-out data...will plot later previously saved results.")

        # ===========================================================================================
        # Interpretations of best model on test data

        if interpret_models == True:

            model_interpretation_start = dt.datetime.now()

            # Fit the preprocessor
            opt_model = pipe.named_steps.regressor

            # fit to and transform train
            X_train_p, X_test_p = get_preprocessed_data(
                pipeline=pipe,
                X_train=X_train,
                X_test=X_test,
                numeric_features=numerical_features_in_data,
                categorical_features=categorical_features_in_data)
            
            print("processed train shape:", X_train_p.shape)
            print("processed test shape:",  X_test_p.shape)
            assert X_train_p.shape[1] == X_test_p.shape[1]

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
                                                random_state=seed, n_jobs=skl_n_jobs, scoring=scoring)
                vars=list(X_test.columns)
                perm_imp_df = pd.DataFrame({
                    'Feature': vars,
                    'importance_mean': result['importances_mean'],
                    'importance_std': result['importances_std']
                })

                # take top n features and then flip so most important at top on graph
                perm_imp_df.sort_values(by="importance_mean", ascending=False, inplace=True, axis=0)     
                save_path_perm_p = Path(perm_imp_save)
                save_path_perm_p.mkdir(parents=True, exist_ok=True)
                filename = f"{run_label}_{model_name}_permutation_importance{run}.csv"
                perm_imp_df.to_csv(save_path_perm_p / filename)

            else:
                filename = f"{load_label}_{model_name}_permutation_importance{run}.csv"
                perm_imp_df = pd.read_csv(perm_imp_load / filename, index_col=[0])
                if plot_nice_names == True:
                    perm_imp_df["Original Feature Name"] = perm_imp_df["Feature"].copy()
                    perm_imp_df["Feature"].replace(var_names_dict, inplace=True)

            # plot
            vars = list(X_test.columns)
            perm_imp_df = perm_imp_df[0:plot_n_features]
            perm_imp_df.sort_values(by="importance_mean", ascending=True, inplace=True, axis=0)

            save_path_perm_plots = Path(perm_imp_save) / "Plots/"
            plot_permutation(perm_imp_df=perm_imp_df,
                            save_path=save_path_perm_plots,
                            save_name=f"{run_label}_{model_name}_permutation{run}")

            if n_permutations > 1:               
                plot_permutation_bars(perm_imp_df=perm_imp_df, save_path=save_path_perm_plots,
                                    save_name=f"{run_label}_{model_name}_MULTIpermutation{run}",
                                    plot_n_features=plot_n_features)

            if run_grouped_permutation == True:
                print("Starting grouped permutation importance")

                save_path_grouped = Path(perm_imp_save) / "Grouped/"
                save_path_grouped.mkdir(parents=True, exist_ok=True)

                result = group_permutation_analysis_avg(X_train, y_train,
                                                    X_test, y_test,
                                                    pipeline=pipe,
                                                    save_path=save_path_grouped,
                                                    model_name=model_name,
                                                    group_dict=block_dict,
                                                    n=n_permutations)

                result_df = pd.DataFrame.from_dict(result, orient='index', columns=["Importance"])
                result_df.sort_values(by="Importance", ascending=False, inplace=True, axis=0)
                
                filename = f"Grouped_{run_label}_{model_name}{run}.csv"
                result_df.to_csv(save_path_grouped / filename)

                # plot grouped permutation importance
                save_path_grouped_plots = save_path_grouped / "Plots"
                save_path_grouped_plots.mkdir(parents=True, exist_ok=True)

                plot_group_perm_importance(result, save_path=save_path_grouped_plots,
                                        save_name=f"Grouped_{run_label}_{model_name}{run}"
                                        )

            # 2) SHAP importance ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            save_path_shap_p = Path(shap_save)
            save_path_shap_p.mkdir(parents=True, exist_ok=True)
            if run_SHAP == True:

                print("Starting SHAP importance for {}".format(model_name))

                # # fit optimised model to transformed train data
                # opt_model.fit(X_train_p, y_train) <- should already be fit from

                # Fit the explainer
                shap_results_dict = {}

                # method_types = ["tree_path_dependent", "interventional"]
                # tree_path_dependent": If features are collinear: Credit is shared or split - which feature gets credit depends on: 
                # -Which one appears higher in the tree
                # -Which one is used more often in splits
                # # (see Lundberg et al., 2020) https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
                method_types = ["interventional"] #interventional: Treats features as if they were independently intervened on 
                                                  # (if two features are highly correlated:Each one gets its own separate credit)
                data_options = [shap.sample(X_test_p, 1000)] # returns: min(K, number_of_rows_in_data) - so if test_data <1000, will return all rows   
                for method_type, data_option in zip(method_types, data_options):
                        # print(method_type)
                        # if (model_name == 'DT'):
                        #     explainer = shap.TreeExplainer(model=opt_model,
                        #                   data=data_option,
                        #                   feature_perturbation=method_type)
                        # else:
                    explainer = shap.KernelExplainer(
                        model=lambda x: opt_model.predict(x),
                        data=data_option)

                    # Calculate the SHAP values and save
                    shap_exp = explainer(X_test_p)
                    shap_values = shap_exp.values

                    # save
                    shap_values_df = pd.DataFrame(shap_values, columns=names)
                    filename = f"{run_label}_SHAP_{model_name}-{method_type}{run}.csv"   
                    shap_values_df.to_csv(save_path_shap_p / filename)

                    shap_results_dict[method_type] = shap_exp

                    filename_pkl = f"{run_label}_SHAP_{model_name}{run}.pkl"
                    file_path_pkl = save_path_shap_p / filename_pkl
                    with open(file_path_pkl, 'wb') as handle:
                        pickle.dump(shap_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            else:
                #load previously calculated SHAP values
                method_type = "interventional" # todo: include "tree path dependent"?
                shap_values_df = pd.read_csv(shap_load + f"{load_label}_SHAP_{model_name}-{method_type}{run}.csv")
                filename_pkl = f"{load_label}_SHAP_{model_name}{run}.pkl"
                file_path_pkl = shap_load / filename_pkl
                with open(file_path_pkl, 'rb') as handle:
                    shap_results_dict = pickle.load(handle)

            # Plot
            if plot_nice_names == True:
                X_test_p_original_names = X_test_p.copy()
                X_test_p.rename(columns=processed_rename_dict, inplace=True)
                names = list(X_test_p.columns)


            shap_plot_save_path = save_path_shap_p / "Plots"
            for method, shap_dict in shap_results_dict.items():
                plot_types = ["bar", "summary", "violin"]
                for plot_type in plot_types:
                    print(plot_type)
                    plot_SHAP(shap_dict, col_list=names, 
                            n_features=plot_n_features, plot_type=plot_type,
                            save_path=shap_plot_save_path, title="SHAP importance (test set)",
                            save_name=f"{run_label}_{model_name}_SHAP_{plot_type}_{method}{run}.png")
                    if plot_type == "summary":
                        plot_SHAP(shap_dict, col_list=names,
                                n_features=plot_n_features, plot_type=None,
                                save_path=shap_plot_save_path, title="SHAP importance (test set)",
                                save_name=f"{run_label}_{model_name}_SHAP_{plot_type}_{method}{run}.png")

            if run_grouped_SHAP == True:
                print('running grouped SHAP importance...')
                # Get extended mapping dict
                shap_save_grouped = Path(shap_save) / "Grouped"
                shap_save_grouped.mkdir(parents=True, exist_ok=True)

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
                filename = f"Grouped_{run_label}_{model_name}_{method_type}{run}.csv"
                group_shap_values_df.to_csv(shap_save_grouped / filename)

                # Plot shap grouped importance
                shap_save_grouped = shap_save_grouped / "Plots"
                shap_save_grouped.mkdir(parents=True, exist_ok=True)
                plot_group_SHAP_importance(group_shap_values,
                                        save_path=shap_save_grouped, save_name=filename)

            model_interpretation_end = dt.datetime.now()
            model_interpretation_time = model_interpretation_end - model_interpretation_start
            print(f"Model interpretation time for {model_name}: {model_interpretation_time}")

    if test_models == True:
        print("Saving all model's test results to file...")
        results_df = pd.DataFrame.from_dict(test_scores)   
        filename = f"all_test_scores_{run_label}{run}.csv"
        results_df.to_csv(all_models_save / filename)

    if test_models == False:
        filename = f"all_test_scores_{load_label}.csv"
        results_df = pd.read_csv(all_models_load / filename, index_col=[0])
        print(f"Loading test results data from: {all_models_save / filename}")

    print("Plotting test performance to compare all models...")

    print(f"Loading test results data from: {all_models_save / filename}")
    x_ticks = ["Prior Ach.", "Prior Ach.+ Track", "Decision Tree", "Random Forest", "Hist. G. Boost.", "XGBoost"]

    save_path_plots = all_models_save / "Plots"
    save_path_plots = Path(save_path_plots)
    save_path_plots.mkdir(parents=True, exist_ok=True)
    plot_results(y="R2", data=results_df, colour='Model',
                save_path=save_path_plots,
                save_name=f"test_summary_all_R2_{run_label}{run}",
                xlab="Prediction Models", ylab="Prediction R Squared",
                title="",
                x_ticks=x_ticks)
    plot_results(y="MAE", data=results_df, colour='Model',
                save_path=save_path_plots,
                save_name=f"test_summary_all_MAE_{run_label}{run}",
                xlab="Prediction Models", ylab="MAE",
                title="",
                x_ticks=x_ticks)
    plot_results(y="RMSE", data=results_df, colour='Model',
                save_path=save_path_plots,
                save_name=f"test_summary_all_RMSE_{run_label}{run}",
                xlab="Prediction Models", ylab="RMSE",
                title="",
                x_ticks=x_ticks)
    # =======================================================================================================
    # end timer 
    script_end = dt.datetime.now()
    run_time = script_end - script_start
    print(f"Script run time: {run_time}")
    print("done!")

if __name__ == "__main__":
    run_pipeline(cfg={})
