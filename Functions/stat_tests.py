import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

def compare_models(
    df,
    model_a,
    model_b,
    decimal_places=3,
    alternative="two-sided",   # "less": A better than B
    dropna=True,
):
    """
    Compare two models using paired tests on absolute errors.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns = signed residuals (y_true - y_pred)
    model_a, model_b : str
        Column names to compare
    alternative : {"less", "greater", "two-sided"}
        Hypothesis for A vs B (less = A has smaller errors)
    dropna : bool
        Drop rows with missing values
    decimal_places : int
        Number of decimal places to round results

    Returns
    -------
    results : dict
        Test statistics and summary metrics
    """

    # extract absolute errors
    ae_a = df[model_a].abs()
    ae_b = df[model_b].abs()

    if dropna:
        valid = ae_a.notna() & ae_b.notna()
        ae_a = ae_a[valid]
        ae_b = ae_b[valid]

    # paired differences
    d = ae_a - ae_b

    # paired t-test
    t_stat, t_p = ttest_rel(ae_a, ae_b, alternative=alternative)

    # Wilcoxon signed-rank test
    w_stat, w_p = wilcoxon(ae_a, ae_b, alternative=alternative)

    results = {
        "n": len(d),
        "mean_MAE_A": round(ae_a.mean(), decimal_places),
        "mean_MAE_B": round(ae_b.mean(), decimal_places),
        "mean_diff": round(d.mean(), decimal_places), # < 0 favors A
        "t_stat": round(t_stat, decimal_places),
        "t_p": round(t_p, decimal_places),
        "wilcoxon_stat": round(w_stat, decimal_places),
        "wilcoxon_p": round(w_p, decimal_places),
    }

    return results
