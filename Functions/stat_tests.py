import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

def compare_models(
    df,
    model_a,
    model_b,
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
        "mean_MAE_A": ae_a.mean(),
        "mean_MAE_B": ae_b.mean(),
        "mean_diff": d.mean(),          # < 0 favors A
        "t_stat": t_stat,
        "t_p": t_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
    }

    return results
