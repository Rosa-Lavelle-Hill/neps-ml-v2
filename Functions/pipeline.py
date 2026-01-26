import numpy as np
import pandas as pd
import xgboost
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from Functions.subsets import cfg_get

# import fixed params from base yaml:
with open("configs/base.yaml", "r") as f:
    cfg_base = yaml.safe_load(f) or {}
seed = cfg_get(cfg_base, ["params", "seed"])
random_state = seed

random_state = seed

def construct_pipelines(numeric_features_index, categorical_features_index):
    """
    Construct multiple sklearn regression pipelines with imputation and encoding.

    Creates preprocessing pipelines for numeric and categorical features,
    including iterative imputation, scaling, and one-hot encoding, and
    combines them with several regression models.

    Parameters
    ----------
    numeric_features_index : list
        Column indices (or names) corresponding to numeric features.
    categorical_features_index : list
        Column indices (or names) corresponding to categorical features.

    Returns
    -------
    tuple of Pipeline
        A tuple containing:
        - ElasticNet pipeline
        - DecisionTreeRegressor pipeline
        - RandomForestRegressor pipeline
        - HistGradientBoostingRegressor pipeline

    Notes
    -----
    - Numeric features are imputed using IterativeImputer with a Bayesian-style model.
    - Categorical features are imputed using IterativeImputer with a RandomForestClassifier.
    - Gradient boosting uses a separate preprocessing pipeline without imputation.
    """
    # define stages of pipeline
    scaler = StandardScaler()
    elastic_net = ElasticNet()
    decision_tree = DecisionTreeRegressor()
    random_forest = RandomForestRegressor()
    oh_encoder = OneHotEncoder(handle_unknown='error', drop="if_binary")
    gradient_boosting = HistGradientBoostingRegressor(validation_fraction=0.1, warm_start=True)

    # Bayesian ridge is quicker/simpler than RF:
    imp_iter_num = IterativeImputer(missing_values=np.nan, max_iter=imputer_max_iter,
                                    random_state=random_state)

    imp_iter_cat = IterativeImputer(estimator=RandomForestClassifier(),
                                    initial_strategy='most_frequent',
                                    missing_values=np.nan,
                                    max_iter=imputer_max_iter,
                                    random_state=random_state,
                                    )
    # Preprocessor pipelines:
    categorical_transformer = Pipeline(
        steps=[("imputer", imp_iter_cat), ("oh_encoder", oh_encoder)]
    )

    numeric_transformer = Pipeline(
        steps=[("imputer", imp_iter_num), ("scaler", scaler)]
    )

    preprocessor = ColumnTransformer(sparse_threshold=0,
        transformers=[
            ("num", numeric_transformer, numeric_features_index),
            ("cat", categorical_transformer, categorical_features_index),
        ]
    )

    # GB preprocessor pipeline:
    categorical_transformer_GB = Pipeline(
        steps=[("oh_encoder", oh_encoder)]
    )

    numeric_transformer_GB = Pipeline(
        steps=[("scaler", scaler)]
    )

    preprocessor_GB = ColumnTransformer(sparse_threshold=0,
        transformers=[
            ("num", numeric_transformer_GB, numeric_features_index),
            ("cat", categorical_transformer_GB, categorical_features_index),
        ]
    )

    # full pipelines:
    pipe_enet = Pipeline([
        ("preprocessor", preprocessor), ('regressor', elastic_net)
    ])

    pipe_dt = Pipeline([
        ("preprocessor", preprocessor), ('regressor', decision_tree)
    ])

    pipe_rf = Pipeline([
        ("preprocessor", preprocessor), ('regressor', random_forest)
    ])

    pipe_gb = Pipeline([
        ("preprocessor", preprocessor_GB), ('regressor', gradient_boosting)
    ])
    return pipe_enet, pipe_dt, pipe_rf, pipe_gb


def construct_pipelines_no_imputation(numeric_features_index, categorical_features_index):
    """
    Construct regression pipelines without missing-value imputation.

    Applies scaling to numeric features and one-hot encoding to categorical
    features, assuming missing values are already handled upstream or
    supported by the estimator.

    Parameters
    ----------
    numeric_features_index : list
        Column indices (or names) corresponding to numeric features.
    categorical_features_index : list
        Column indices (or names) corresponding to categorical features.

    Returns
    -------
    tuple of Pipeline
        A tuple containing:
        - ElasticNet pipeline
        - DecisionTreeRegressor pipeline
        - RandomForestRegressor pipeline
        - HistGradientBoostingRegressor pipeline

    Notes
    -----
    - No imputation is performed.
    - OneHotEncoder raises an error for unseen categories.
    """
    # define stages of pipeline
    scaler = StandardScaler()
    elastic_net = ElasticNet()
    random_forest = RandomForestRegressor()
    decision_tree = DecisionTreeRegressor()
    oh_encoder = OneHotEncoder(handle_unknown='error', drop="if_binary")
    gradient_boosting = HistGradientBoostingRegressor(validation_fraction=0.1, warm_start=True)

    # Preprocessor pipelines (no imputation):
    # one hot (oh) encoding):
    categorical_transformer = Pipeline(
        steps=[("oh_encoder", oh_encoder)]
    )

    numeric_transformer = Pipeline(
        steps=[("scaler", scaler)]
    )

    preprocessor = ColumnTransformer(sparse_threshold=0,
        transformers=[
            ("num", numeric_transformer, numeric_features_index),
            ("cat", categorical_transformer, categorical_features_index),
        ]
    )

    # full pipelines:
    pipe_enet = Pipeline([
        ("preprocessor", preprocessor), ('regressor', elastic_net)
    ])

    pipe_dt = Pipeline([
        ("preprocessor", preprocessor), ('regressor', decision_tree)
    ])

    pipe_rf = Pipeline([
        ("preprocessor", preprocessor), ('regressor', random_forest)
    ])

    pipe_gb = Pipeline([
        ("preprocessor", preprocessor), ('regressor', gradient_boosting)
    ])
    return pipe_enet, pipe_dt, pipe_rf, pipe_gb


def construct_pipelines_all_no_imputation(categorical_features_index):
    """
    Construct regression pipelines with no imputation and minimal preprocessing.

    Only categorical variables are one-hot encoded. Numeric variables are passed
    directly to the model without scaling or imputation.

    Parameters
    ----------
    numeric_features_index : list
        Column indices (or names) corresponding to numeric features.
    categorical_features_index : list
        Column indices (or names) corresponding to categorical features.

    Returns
    -------
    tuple of Pipeline
        A tuple containing:
        - DecisionTreeRegressor pipeline (sklearn)
        - Random forest pipeline using XGBoost's XGBRFRegressor
        - HistGradientBoostingRegressor pipeline
        - XGBoost regression pipeline (XGBRegressor)

    Notes
    -----
    - Designed for models that handle missing values natively (e.g. XGBoost).
    - Numeric features are passed through unchanged.
    - Categorical features are one-hot encoded with unknown categories ignored.
    """

    # Define preprocessing steps for categorical features only
    oh_encoder = OneHotEncoder(handle_unknown='ignore', drop="if_binary")  # Adjusted to ignore unknown categories

    # Preprocessor pipelines
    categorical_transformer = Pipeline(steps=[("oh_encoder", oh_encoder)])

    # ColumnTransformer: Only preprocess categorical features
    preprocessor = ColumnTransformer(
        sparse_threshold=0,
        transformers=[
            ("cat", categorical_transformer, categorical_features_index)
        ],
        remainder="passthrough"  # Pass numeric features directly to the model without processing
    )

    # Define regressors
    # sklearn:
    decision_tree = DecisionTreeRegressor()
    hist_gradient_boosting = HistGradientBoostingRegressor(validation_fraction=0.1, warm_start=True)
    # xgboost package:
    random_forest = xgboost.XGBRFRegressor(booster="gbtree", objective="reg:squarederror",
                                           subsample=0.8,
                                           learning_rate=1) # set to 1 for random forest
    # see here for rf using xgboost package: https://xgboost.readthedocs.io/en/latest/tutorials/rf.html
    xgboost_regressor = xgboost.XGBRegressor(booster="gbtree",
                                             tree_method="hist", objective="reg:squarederror",
                                             subsample=0.8, # % of rows sampled
                                             colsample_bynode=0.3, # akin to mtry/max_features
                                             missing=np.nan, # Explicitly set missing value
                                             max_bin=200, # speeds up computation time from 256 (default) = bins for continuous variables
                                             num_parallel_tree=1) # set to one otherwise a boosted random forest
    # Full pipelines
    # sklearn:
    pipe_dt = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", decision_tree)
    ])

    pipe_hgb = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", hist_gradient_boosting)
    ])

    # xgboost package:
    pipe_xgb = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", xgboost_regressor)
    ])

    pipe_rf = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", random_forest)
    ])

    return pipe_dt, pipe_rf, pipe_hgb, pipe_xgb



def get_preprocessed_col_names(X, pipe, cat_vars):
    """
    Retrieve feature names after preprocessing from a fitted pipeline.

    Extracts the names of numeric features and the expanded one-hot encoded
    categorical features in the correct post-transformation order.

    Parameters
    ----------
    X : pandas.DataFrame
        Original input data before preprocessing.
    pipe : Pipeline
        Fitted sklearn pipeline containing a ColumnTransformer named 'preprocessor'.
    cat_vars : list
        List of categorical feature names.

    Returns
    -------
    list
        List of feature names corresponding to the transformed design matrix.

    Notes
    -----
    - Assumes categorical features are transformed using OneHotEncoder.
    - Assumes categorical transformer is the second transformer in the ColumnTransformer.
    """
    numeric_features = X.drop(cat_vars, inplace=False, axis=1).columns
    dum_names = list(pipe.named_steps['preprocessor'].transformers_[1][1].named_steps['oh_encoder']
                     .get_feature_names_out(cat_vars))
    num_names = list(X[numeric_features].columns.values)
    names = num_names + dum_names
    return names


def get_preprocessed_data(pipeline, X_train, X_test, numeric_features, categorical_features):
    """
    Extract preprocessed training and test data from a pipeline.

    Fits the preprocessing steps of a pipeline on the training data,
    applies them to both training and test sets, and returns the results
    as pandas DataFrames with correct feature names.

    Parameters
    ----------
    pipeline : Pipeline
        Sklearn pipeline containing a 'preprocessor' step.
    X_train : pandas.DataFrame
        Training data before preprocessing.
    X_test : pandas.DataFrame
        Test data before preprocessing.
    numeric_features : list
        List of numeric feature names.
    categorical_features : list
        List of categorical feature names.

    Returns
    -------
    pandas.DataFrame
        Preprocessed training data with named columns.
    pandas.DataFrame
        Preprocessed test data with named columns.

    Notes
    -----
    - Categorical features are assumed to be one-hot encoded.
    - Saves the preprocessed training data to disk for inspection.
    - Assumes categorical transformer is the first transformer in the ColumnTransformer.
    """
    # Extract the preprocessor from the pipeline
    preprocessor = pipeline.named_steps["preprocessor"]

    # Fit the preprocessor on X_train and transform X_train and X_test
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Get column names from the preprocessor
    # Categorical features are one-hot encoded; numeric features are passed through
    categorical_feature_names = preprocessor.transformers_[0][1] \
        .named_steps["oh_encoder"].get_feature_names_out(categorical_features)
    all_feature_names = list(categorical_feature_names) + numeric_features

    # Convert the preprocessed data back to DataFrames with column names
    X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed, columns=all_feature_names, index=X_train.index)
    X_train_preprocessed_df.to_csv("Data/Preprocessed/X_train_p.csv") #  save for original transformed feature names
    X_test_preprocessed_df = pd.DataFrame(X_test_preprocessed, columns=all_feature_names, index=X_test.index)

    return X_train_preprocessed_df, X_test_preprocessed_df






