def get_param_grids(seed: int):

    # # sklearn decision tree:
    # dt_param_grid = {"regressor__min_samples_split": [2, 3, 4, 5],
    #              "regressor__max_depth": [10, 15, 20],
    #              "regressor__random_state": [seed],
    #              "regressor__max_features": [0.3, 0.4, 0.5, 0.6]}

    dt_param_grid = {"regressor__min_samples_split": [5, 8],
                "regressor__max_depth": [10],
                "regressor__random_state": [seed],
                "regressor__max_features": [0.6, 0.8]}

    #  sklearn histogram-based gradient boosting:
    # hgb_param_grid = {"regressor__min_samples_leaf": [5, 10, 20],
    #              "regressor__max_depth": [10, 20, 30],
    #              "regressor__max_iter": [300, 500, 700],
    #              "regressor__random_state": [seed],
    #              # "regressor__max_features": [0.3, 0.5, 0.7],
    #              "regressor__learning_rate": [0.3, 0.1, 0.001]}

    hgb_param_grid = {"regressor__min_samples_leaf": [10],
                "regressor__max_depth": [20], # The maximum depth of each tree. The depth of a tree is the number of edges to go from the root to the deepest leaf. 
                "regressor__max_iter": [250],  # The maximum number of iterations of the boosting process, i.e. the maximum number of trees.
                "regressor__random_state": [seed],
                "regressor__max_features": [0.3, 0.5, 0.7],  #Proportion of randomly chosen features in each and every node split. This is a form of regularization, smaller values make the trees weaker learners and might prevent overfitting.
                "regressor__learning_rate": [0.1, 0.3], #The learning rate, also known as shrinkage. This is used as a multiplicative factor for the leaves values. Use 1 for no shrinkage.
                "regressor__l2_regularization": [0.0, 0.2]} # The L2 regularization parameter penalizing leaves with small hessians. 0 for no regularization.

    # -----------
    # xgboost random forest:
    # rf_param_grid = {"regressor__max_depth": [10, 15, 20],
    #                  "regressor__n_estimators": [300, 500, 700], 
    #                  "regressor__colsample_bynode" : [0.4, 0.5, 0.6, 0.7],
    #                  "regressor__random_state": [seed]}

    rf_param_grid = {"regressor__max_depth": [10],
                    "regressor__n_estimators": [300], 
                    "regressor__colsample_bynode" : [0.4],
                    "regressor__random_state": [seed]}

    # xgboost xgboost:
    # xgb_param_grid = {"regressor__max_depth": [4, 6, 8], # The maximum depth of a tree. 
    #                  "regressor__random_state": [seed],
    #                  "regressor__learning_rate": [0.1, 0.001], #Boosting learning rate (xgb’s “eta”). Step size shrinkage used in update to prevent overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.
    #                  "regressor__alpha": [0, 0.2, 0.5], #L1 regularization term on weights. Increasing this value will make model more conservative.
    #                  "regressor__max_leaves": [0], # Maximum number of leaves (nodes to be added); 0 indicates no limit.
    #                  "regressor__n_estimators": [300, 500, 700]} # Number of gradient boosted trees. Equivalent to number of boosting rounds.

    xgb_param_grid = {"regressor__max_depth": [4],
                    "regressor__random_state": [seed],
                    "regressor__learning_rate": [0.1, 0.001], 
                    "regressor__alpha": [0], 
                    "regressor__max_leaves": [0], 
                    "regressor__n_estimators": [300]}

    # find more info on xgboost hyper-paramters here: https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn

    # CatBoost (random_seed is set at construction; only tunable params passed here)
    cat_param_grid = {
        "regressor__iterations": [300],
        "regressor__learning_rate": [0.1, 0.05],
        "regressor__depth": [4, 6],
        "regressor__l2_leaf_reg": [1, 3],
    }

    return [dt_param_grid, rf_param_grid, hgb_param_grid, xgb_param_grid, cat_param_grid]
# ------------------------------------------------------------------
# test grids:

def get_test_param_grids(seed: int):
    test_dt_param_grid = {'regressor__max_depth': [5],
                'regressor__min_samples_split': [2],
                'regressor__max_features': [0.3],
                'regressor__random_state': [93]}

    test_rf_param_grid ={"regressor__max_depth": [5],
                    "regressor__num_parallel_tree": [50],
                    "regressor__colsample_bynode" : [0.2],
                    "regressor__random_state": [seed]}

    test_hgb_param_grid = {"regressor__min_samples_leaf": [5],
                        "regressor__max_depth": [5],
                        "regressor__random_state": [93],
                        "regressor__max_iter": [50],
                        "regressor__learning_rate": [0.1]}

    test_xgb_param_grid = {"regressor__max_depth": [5],
                    "regressor__num_parallel_tree": [50],
                    "regressor__random_state": [seed],
                    "regressor__learning_rate": [0.1]}

    test_cat_param_grid = {
        "regressor__iterations": [50],
        "regressor__learning_rate": [0.1],
        "regressor__depth": [4],
    }
    return [test_dt_param_grid, test_rf_param_grid, test_hgb_param_grid, test_xgb_param_grid, test_cat_param_grid]

# ------------------------------------------------------------------