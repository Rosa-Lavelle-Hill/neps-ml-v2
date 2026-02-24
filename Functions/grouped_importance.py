import yaml
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from Functions.subsets import cfg_get
# import fixed params from base yaml:

with open("configs/base.yaml", "r") as f:
    cfg_base = yaml.safe_load(f) or {}
seed = cfg_get(cfg_base, ["params", "seed"])
random_state = seed

def group_permutation_analysis(X_train, y_train, X_test, y_test,
                               pipeline, group_dict, model_name,
                               save_path,
                               scoring=r2_score, random_state=seed):
    """
    Perform group permutation analysis.

    Parameters:
        pipeline (Pipeline): The sklearn pipeline with preprocessing and a prediction model.
        group_dict (dict): Dictionary mapping column names to group names.
        scoring (callable): Scoring function that takes y_true, y_pred.

    Returns:
        dict: Group importance scores (difference in performance after permutation).
    """

    group_dict = {col: group for col, group in group_dict.items() if col in X_train.columns}

    # Fit pipeline on original data
    pipeline.fit(X_train, y_train)
    original_score = scoring(y_test, pipeline.predict(X_test))
    # todo: 0.02 better -- because not one hot coded?

    # Map columns to groups
    group_to_columns = {}
    for col, group in group_dict.items():
        group_to_columns.setdefault(group, []).append(col)

    # Group permutation analysis
    group_importance = {}
    rng = np.random.default_rng(random_state)
    for group, columns in group_to_columns.items():
        # Permute group columns
        X_test_permuted = X_test.copy()
        for col in columns:
            X_test_permuted[col] = rng.permutation(X_test_permuted[col])

        # Evaluate performance on permuted data
        permuted_score = scoring(y_test, pipeline.predict(X_test_permuted))
        group_importance[group] = round(original_score - permuted_score, 3)  # Importance is score difference

    return group_importance


import numpy as np
from sklearn.metrics import r2_score

def group_permutation_analysis_avg(X_train, y_train, X_test, y_test,
                                   pipeline, group_dict, model_name,
                                   save_path, n,
                                   scoring=r2_score, random_state=None):
    """
    Perform averaged group permutation analysis.

    Parameters:
        pipeline (Pipeline): The sklearn pipeline with preprocessing and a prediction model.
        group_dict (dict): Dictionary mapping column names to group names.
        n (int): Number of permutations to average.
        scoring (callable): Scoring function that takes y_true, y_pred.
        random_state (int or None): Random seed for reproducibility.

    Returns:
        dict: Average group importance scores.
    """

    group_dict = {col: group for col, group in group_dict.items() if col in X_train.columns}

    # Fit pipeline on original data
    pipeline.fit(X_train, y_train)
    original_score = scoring(y_test, pipeline.predict(X_test))

    # Map columns to groups
    group_to_columns = {}
    for col, group in group_dict.items():
        group_to_columns.setdefault(group, []).append(col)

    # Initialize results
    group_importance_avg = {group: 0.0 for group in group_to_columns}
    rng = np.random.default_rng(random_state)

    for _ in range(n):
        for group, columns in group_to_columns.items():
            # Permute group columns
            X_test_permuted = X_test.copy()
            for col in columns:
                X_test_permuted[col] = rng.permutation(X_test_permuted[col])

            # Evaluate performance on permuted data
            permuted_score = scoring(y_test, pipeline.predict(X_test_permuted))
            group_importance_avg[group] += original_score - permuted_score

    # Average results
    group_importance_avg = {group: round(score / n, 3) for group, score in group_importance_avg.items()}

    return group_importance_avg




