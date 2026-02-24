import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
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
        "MAE_A": round(ae_a.mean(), decimal_places),
        "MAE_B": round(ae_b.mean(), decimal_places),
        "mean_diff": round(d.mean(), decimal_places), # < 0 favors A
        "t_stat": round(t_stat, decimal_places),
        "t_p": round(t_p, decimal_places),
        "wilcoxon_stat": round(w_stat, decimal_places),
        "wilcoxon_p": round(w_p, decimal_places),
    }

    return results



def plot_error_distributions(
    df,
    model_a,
    model_b,
    save_path,
    bins=40,
    alpha=0.4
):
    """
    Overlay distributions of absolute errors for two models.
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns = signed residuals (y_true - y_pred)
    model_a, model_b : str
        Column names to compare
    save_path : str
        Path to save the plot
    bins : int
        Number of bins for histogram
    alpha : float
        Transparency for histogram bars (0 to 1)
    Returns
    -------
    None    
    """

    ae_a = df[model_a].abs()
    ae_b = df[model_b].abs()

    # align rows
    valid = ae_a.notna() & ae_b.notna()
    ae_a = ae_a[valid]
    ae_b = ae_b[valid]

    plt.figure(figsize=(7, 5))

    sns.histplot(
        ae_a,
        bins=bins,
        stat="density",
        alpha=alpha,
        label=model_a,
        kde=True
    )

    sns.histplot(
        ae_b,
        bins=bins,
        stat="density",
        alpha=alpha,
        label=model_b,
        kde=True
    )

    plt.xlabel("Absolute error")
    plt.ylabel("Density")
    plt.title("{model_a} vs. {model_b}".format(
        model_a=model_a,
        model_b=model_b), fontsize=10)
    plt.savefig(f"{save_path}/error_distributions_{model_a}_vs_{model_b}.png")
    plt.close()
    return


def plot_difference_normality(
    df,
    model_a,
    model_b,
    save_path,
    bins=40,
    show_shapiro=True,
    
):
    """
    Plot diagnostics for normality of paired differences in absolute errors.

    Parameters
    ----------
    df : pd.DataFrame
        Columns contain signed residuals (y_true - y_pred)
    model_a, model_b : str
        Column names to compare
    bins : int
        Histogram bins
    show_shapiro : bool
        Whether to run Shapiro–Wilk test (diagnostic only)
    save_path : str
        Path to save the plot

    Returns
    -------
    None
    """

    # absolute errors
    ae_a = df[model_a].abs()
    ae_b = df[model_b].abs()

    # align rows
    valid = ae_a.notna() & ae_b.notna()
    ae_a = ae_a[valid]
    ae_b = ae_b[valid]

    # paired differences
    d = ae_a - ae_b

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # -----------------------------
    # Histogram + KDE
    # -----------------------------
    sns.histplot(d, bins=bins, kde=True, ax=axes[0])
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].set_title("Distribution of paired differences")
    axes[0].set_xlabel(f"|{model_a}| − |{model_b}| absolute error")

    # -----------------------------
    # Q–Q plot
    # -----------------------------
    stats.probplot(d, dist="norm", plot=axes[1])
    axes[1].set_title("Q–Q plot of paired differences")

    plt.tight_layout()
    plt.savefig(f"{save_path}/difference_normality_{model_a}_vs_{model_b}.png")
    plt.close()

    # -----------------------------
    # Optional Shapiro–Wilk
    # -----------------------------
    if show_shapiro:
        W, p = stats.shapiro(d)
        print(
            f"Shapiro–Wilk test: W = {W:.3f}, p = {p:.4f} "
            "(diagnostic only; sensitive to n)"
        )

    return 
