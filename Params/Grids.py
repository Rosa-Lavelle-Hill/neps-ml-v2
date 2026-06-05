def get_param_grids(seed: int):

    # sklearn decision tree:
    dt_param_grid = {"regressor__min_samples_split": [3, 5, 8], # The minimum number of samples required to split an internal node. This is a regularization parameter that can help prevent overfitting. Higher values will result in fewer splits and a simpler tree, while lower values will allow for more splits and a more complex tree.
                "regressor__max_depth": [5, 10, 15], # The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
                "regressor__random_state": [seed], # The seed is set for reproducibility, ensuring that the same random processes (like feature selection or data shuffling) yield the same results across different runs. 
                "regressor__max_features": [0.3, 0.5, 0.7]} # Proportion of randomly chosen features in each and every node split. This is a form of regularization, smaller values make the trees weaker learners and might prevent overfitting.

    # sklearn histogram-based gradient boosting:
    hgb_param_grid = {"regressor__min_samples_leaf": [5, 10, 20], # The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.
                "regressor__max_depth": [5, 10, 20], # The maximum depth of each tree. The depth of a tree is the number of edges to go from the root to the deepest leaf.
                "regressor__max_iter": [200, 350, 500],  # The maximum number of iterations of the boosting process, i.e. the maximum number of trees.
                "regressor__random_state": [seed],
                "regressor__max_features": [0.3, 0.5, 0.7],  #Proportion of randomly chosen features in each and every node split. This is a form of regularization, smaller values make the trees weaker learners and might prevent overfitting.
                "regressor__learning_rate": [0.3, 0.1], #The learning rate, also known as shrinkage. This is used as a multiplicative factor for the leaves values. Use 1 for no shrinkage.
                "regressor__l2_regularization": [0.0, 0.2]} # The L2 regularization parameter penalizing leaves with small hessians. 0 for no regularization.

    # -----------
    # xgboost random forest:
    rf_param_grid = {"regressor__max_depth": [8, 12, 15],  # The maximum depth of a tree.
                    "regressor__n_estimators": [200, 350, 500],  # The number of trees in the forest.
                    "regressor__colsample_bynode" : [0.4, 0.5, 0.6, 0.7],  # The fraction of features to consider for each split.
                    "regressor__random_state": [seed]}  # The seed is set for reproducibility.

    # xgboost xgboost:
    xgb_param_grid = {"regressor__max_depth": [4, 6, 8], # The maximum depth of a tree.
                     "regressor__random_state": [seed],
                     "regressor__learning_rate": [0.3, 0.1], #Boosting learning rate (xgb's "eta"). Step size shrinkage used in update to prevent overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.
                     "regressor__alpha": [0, 0.2, 0.5], #L1 regularization term on weights. Increasing this value will make model more conservative.
                     "regressor__max_leaves": [0], # Maximum number of leaves (nodes to be added); 0 indicates no limit.
                     "regressor__n_estimators": [200, 350, 500]} # Number of gradient boosted trees. Equivalent to number of boosting rounds.
                    # find more info on xgboost hyper-paramters here: https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn

    # CatBoost
    cat_param_grid = {
        "regressor__iterations": [300, 500], # The maximum number of trees that can be built when solving machine learning problems. This is a key parameter for controlling the complexity of the model and preventing overfitting. A higher number of iterations can lead to a more complex model, while a lower number can result in underfitting.
        "regressor__learning_rate": [0.1, 0.05], # The learning rate, also known as shrinkage. This is used as a multiplicative factor for the leaves values. Use 1 for no shrinkage. A smaller learning rate makes the model more robust to overfitting but requires more iterations to converge.
        "regressor__depth": [4, 6], # The depth of the tree. This is a key parameter for controlling the complexity of the model and preventing overfitting. A higher depth can lead to a more complex model, while a lower depth can result in underfitting.
        "regressor__l2_leaf_reg": [1, 3], # L2 regularization term on weights. Increasing this value will make model more conservative.
        "regressor__rsm": [0.5, 0.7], # Random subspace method (feature sampling ratio). Similar role to max_features: controls the fraction of features considered at each split, which can improve generalization and speed.
    }
        # (random_seed is set at construction)

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