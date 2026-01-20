
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score

def diagnose_degenerate_features(
    result, X, sampler=None, feature_names=None,
    eps_var=1e-12, r2_threshold=0.999, acc_threshold=0.999,
    n_copies_check=200, random_state=0
):
    """
    result: DataFrame from rpt.summarize(), must contain ['estimate','std_error','p_value'].
            Its index or a column must identify the feature (the 'removal').
    X:      original features used by the learner (DataFrame is best; ndarray ok if feature_names provided).
    sampler: optional sampler object used by your CIT; if it has .sample(), we test resample variance.
    """

    # --- figure out the feature label column in `result` ---
    # If the index holds the feature name/id, use it; otherwise try the first column.
    res = result.copy()
    if res.index.name in (None, "") and "removal" not in res.columns:
        # try to use the first column (like "Unnamed: 0")
        res = res.reset_index()
        feature_col = res.columns[0]
    elif "removal" in res.columns:
        feature_col = "removal"
    else:
        # index holds the feature name
        res = res.reset_index().rename(columns={"index": "removal"})
        feature_col = "removal"

    # Ensure we can map to X's columns
    if isinstance(X, np.ndarray):
        if feature_names is None:
            raise ValueError("When X is ndarray, please pass feature_names=list/array of column labels.")
        X_df = pd.DataFrame(X, columns=list(feature_names))
    else:
        X_df = X.copy()

    # Focus on rows with zero std_error
    zero_se = res[res["std_error"] == 0].copy()
    if zero_se.empty:
        print("No features with std_error == 0. Nothing to diagnose.")
        return pd.DataFrame()

    out_rows = []
    for _, row in zero_se.iterrows():
        feat = row[feature_col]
        # robustly resolve column name/index to actual column in X_df
        if isinstance(feat, (int, np.integer)) and feat in range(X_df.shape[1]):
            feat_name = X_df.columns[int(feat)]
        else:
            feat_name = feat

        # Skip if feature not found
        if feat_name not in X_df.columns:
            out_rows.append({
                "feature": feat, "found_in_X": False,
                "feature_var": np.nan, "n_unique": np.nan,
                "cv_score": np.nan, "predictability_type": None,
                "sampler_avg_copy_var": np.nan,
                "estimate": row.get("estimate", np.nan),
                "p_value": row.get("p_value", np.nan),
                "reason": "feature not in X"
            })
            continue

        xj = X_df[feat_name].values
        X_minus = X_df.drop(columns=[feat_name]).values

        # 1) Constant / near-constant?
        feature_var = float(np.var(xj))
        n_unique = int(pd.Series(xj).nunique(dropna=False))
        is_constant = (feature_var <= eps_var) or (n_unique <= 1)

        # 2) Predictability from others (quick holdout)
        # Decide numeric vs "binary-looking" categorical
        is_binary = (n_unique == 2)

        cv_score = np.nan
        predictability_type = None

        if X_minus.shape[1] > 0 and not is_constant:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_minus, xj, test_size=0.33, random_state=random_state
            )

            if is_binary:
                # encode labels if not numeric {0,1}
                le = LabelEncoder()
                y_tr_enc = le.fit_transform(y_tr)
                y_te_enc = le.transform(y_te)
                clf = RandomForestClassifier(
                    n_estimators=200, random_state=random_state, n_jobs=-1
                )
                clf.fit(X_tr, y_tr_enc)
                y_hat = clf.predict(X_te)
                cv_score = accuracy_score(y_te_enc, y_hat)
                predictability_type = "accuracy"
            else:
                # numeric (or multi-valued categorical treated as numeric)
                # try a non-linear regressor to catch complex predictability
                reg = RandomForestRegressor(
                    n_estimators=200, random_state=random_state, n_jobs=-1
                )
                reg.fit(X_tr, y_tr)
                y_hat = reg.predict(X_te)
                cv_score = r2_score(y_te, y_hat)
                predictability_type = "r2"

        # 3) Sampler degeneracy check (optional)
        sampler_avg_copy_var = np.nan
        if sampler is not None and hasattr(sampler, "sample") and X_minus.shape[1] > 0:
            try:
                copies = sampler.sample(
                    X_minus, xj, n_copies=n_copies_check, random_state=random_state
                )  # expected shape (n_copies, n_samples)
                # variance across copies for each sample, then average
                samplewise_var = np.var(copies, axis=0)
                sampler_avg_copy_var = float(np.mean(samplewise_var))
            except Exception as e:
                sampler_avg_copy_var = np.nan  # keep going gracefully

        # Decide reason
        reasons = []
        if is_constant:
            reasons.append("constant/near-constant")
        if predictability_type == "r2" and (not np.isnan(cv_score)) and cv_score >= r2_threshold:
            reasons.append(f"near-perfectly predictable (R2≈{cv_score:.3f})")
        if predictability_type == "accuracy" and (not np.isnan(cv_score)) and cv_score >= acc_threshold:
            reasons.append(f"near-perfectly predictable (acc≈{cv_score:.3f})")
        if not np.isnan(sampler_avg_copy_var) and sampler_avg_copy_var <= eps_var:
            reasons.append("sampler produces ~identical copies (degenerate null)")

        if not reasons:
            reasons.append("unknown (numerical precision or edge case)")

        out_rows.append({
            "feature": feat_name,
            "found_in_X": True,
            "feature_var": feature_var,
            "n_unique": n_unique,
            "cv_score": cv_score,
            "predictability_type": predictability_type,
            "sampler_avg_copy_var": sampler_avg_copy_var,
            "estimate": row.get("estimate", np.nan),
            "p_value": row.get("p_value", np.nan),
            "reason": "; ".join(reasons)
        })

    out = pd.DataFrame(out_rows).sort_values(by=["reason","feature"]).reset_index(drop=True)
    return out


