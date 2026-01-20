from fixed_params import seed
# # sklearn:
# dt_param_grid = {"regressor__min_samples_split": [2, 3, 4, 5],
#              "regressor__max_depth": [10, 15, 20],
#              "regressor__random_state": [seed],
#              "regressor__max_features": [0.3, 0.4, 0.5, 0.6]}
dt_param_grid = {"regressor__min_samples_split": [5, 8],
             "regressor__max_depth": [10],
             "regressor__random_state": [seed],
             "regressor__max_features": [0.6, 0.8]}

enet_param_grid = {"regressor__tol": [0.001, 0.01],
               "regressor__max_iter": [1000],
               "regressor__l1_ratio": [0.1, 0.25, 0.5, 0.75, 1],
               "regressor__alpha": [0.1, 0.2, 0.5, 1, 1.5, 2]}

# rf_param_grid = {"regressor__min_samples_split": [2, 3, 4, 5],
#              "regressor__max_depth": [10, 15, 20],
#              "regressor__n_estimators": [300, 500, 700],
#              "regressor__random_state": [seed],
#              "regressor__max_features": [0.3, 0.4, 0.5, 0.6]}

# hgb_param_grid = {"regressor__min_samples_leaf": [5, 10, 20],
#              "regressor__max_depth": [10, 20, 30],
#              "regressor__max_iter": [250],
#              "regressor__random_state": [seed],
#              # "regressor__max_features": [0.3, 0.5],
#              "regressor__learning_rate": [0.1, 0.001]}

hgb_param_grid = {"regressor__min_samples_leaf": [10],
             "regressor__max_depth": [20],
             "regressor__max_iter": [250],  # The maximum number of iterations of the boosting process, i.e. the maximum number of trees.
             "regressor__random_state": [seed],
             "regressor__max_features": [0.3, 0.5, 0.7],  #Proportion of randomly chosen features in each and every node split. This is a form of regularization, smaller values make the trees weaker learners and might prevent overfitting.
             "regressor__learning_rate": [0.1, 0.3]} #The learning rate, also known as shrinkage. This is used as a multiplicative factor for the leaves values. Use 1 for no shrinkage.
# todo: option for l2 regularisation
# -----------
# xgboost rf:
# rf_param_grid = {"regressor__max_depth": [10, 15, 20],
#                  "regressor__num_parallel_tree": [300, 500],
#                  "regressor__colsample_bynode" : [0.2, 0.4, 0.6],
#                  "regressor__random_state": [seed]}
rf_param_grid = {"regressor__max_depth": [20],
                 "regressor__n_estimators": [300], # change to 700 later
                 "regressor__colsample_bynode" : [0.6],
                 "regressor__random_state": [seed]}
# todo: look into other params!!!!
# xgboost xgb:
xgb_param_grid = {"regressor__max_depth": [4, 6, 8],
                 "regressor__random_state": [seed],
                 "regressor__learning_rate": [0.1, 0.001], #Boosting learning rate (xgb’s “eta”). Step size shrinkage used in update to prevent overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.
                 "regressor__alpha": [0, 0.2, 0.5], #L1 regularization term on weights. Increasing this value will make model more conservative.
                 "regressor__max_leaves": [0], # Maximum number of leaves (nodes to be added); 0 indicates no limit.
                 "regressor__n_estimators": [300]} # change to 700 later. Number of gradient boosted trees. Equivalent to number of boosting rounds.

# find more info on xgboost hyper-paramters here: https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn
# ------------------------------------------------------------------
# test grids:
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

test_enet_param_grid = {"regressor__tol": [0.1],
               "regressor__max_iter": [1],
               "regressor__l1_ratio": [0.5],
               "regressor__alpha": [0.5]}
# ------------------------------------------------------------------
# # old test grids:
# test_dt_param_grid = {'regressor__max_depth': [5],
#              'regressor__min_samples_split': [2],
#              'regressor__max_features': [0.3],
#              'regressor__random_state': [93]}
#
# test_rf_param_grid = {'regressor__max_depth': [5],
#              'regressor__max_features': [0.3],
#              'regressor__min_samples_split': [2],
#              'regressor__n_estimators': [5],
#              'regressor__random_state': [93]}
#
# test_gb_param_grid = {"regressor__min_samples_leaf": [5],
#                      "regressor__max_depth": [5],
#                      "regressor__random_state": [93],
#                      "regressor__max_iter": [50],
#                      "regressor__learning_rate": [0.1]}
#
# test_enet_param_grid = {"regressor__tol": [0.1],
#                "regressor__max_iter": [1],
#                "regressor__l1_ratio": [0.5],
#                "regressor__alpha": [0.5]}
# ------------------------------------------------------------------
# sim grids
sim_dt_param_grid = {"min_samples_split": [2, 4, 6],
             "max_depth": [5, 10, 20],
             "random_state": [93],
             "max_features": [1]}

sim_rf_param_grid = {"min_samples_split": [2, 4],
             "max_depth": [10, 15, 20],
             "n_estimators": [300, 500],
             "random_state": [93],
             "max_features": [0.3, 0.5]}

sim_gb_param_grid = {"min_samples_leaf": [5, 10, 20],
             "max_depth": [10, 20, 30],
             "max_iter": [300, 500],
             "random_state": [93],
             # "regressor__max_features": [0.3, 0.5],
             "learning_rate": [0.1, 0.001]}