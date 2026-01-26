import pandas as pd
from pathlib import Path 

def load_var_info(csv_path: str) -> pd.DataFrame:
    """
    Load variable metadata from a CSV and normalize key string columns.

    Reads a CSV file into a DataFrame and strips whitespace from the 'var'
    and 'Variable_Group' columns to reduce mismatches due to formatting.

    Parameters
    ----------
    csv_path : str
        Path to the variable-info CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded variable info with normalized 'var' and 'Variable_Group' columns.

    Raises
    ------
    FileNotFoundError
        If `csv_path` does not exist.
    KeyError
        If required columns ('var', 'Variable_Group') are missing.
    """
    df = pd.read_csv(csv_path)
    # normalize strings a bit (optional but helps)
    df["var"] = df["var"].astype(str).str.strip()
    df["Variable_Group"] = df["Variable_Group"].astype(str).str.strip()
    return df

def columns_for_groups(
    var_info: pd.DataFrame,
    include_groups=None,
    exclude_groups=None,
    var_col: str = "var",
    group_col: str = "Variable_Group",):
    """
    Compute a sorted list of variable names belonging to selected groups.

    Selects variables from `var_info` based on group inclusion/exclusion rules.
    If `include_groups` is provided, only variables in those groups are included.
    If `exclude_groups` is provided, variables in those groups are removed.

    Parameters
    ----------
    var_info : pandas.DataFrame
        Variable metadata table containing at least `var_col` and `group_col`.
    include_groups : list, set, or None, optional
        Groups to include. If None or empty, start from all variables.
    exclude_groups : list, set, or None, optional
        Groups to exclude from the selected set.
    var_col : str, default="var"
        Column in `var_info` containing variable names.
    group_col : str, default="Variable_Group"
        Column in `var_info` containing group labels.

    Returns
    -------
    list of str
        Sorted list of cleaned variable names.

    Notes
    -----
    - Drops names that are empty/blank, non-strings, or equal to 'nan' (case-insensitive).
    """
    # Start set of columns
    if include_groups:
        cols = set(var_info.loc[var_info[group_col].isin(include_groups), var_col])
    else:
        cols = set(var_info[var_col])

    if exclude_groups:
        cols -= set(var_info.loc[var_info[group_col].isin(exclude_groups), var_col])

    # drop missing/blank column names if any
    cols = {c for c in cols if isinstance(c, str) and c.strip() and c.lower() != "nan"}
    return sorted(cols)

def apply_subset(X, var_info: pd.DataFrame, subset_cfg: dict):
    """
    Subset columns of X according to a variable-group configuration.

    Uses `subset_cfg` to select variables based on group inclusion/exclusion
    in `var_info`, then returns X restricted to columns that exist in X.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix to subset (columns are variables).
    var_info : pandas.DataFrame
        Variable metadata table used to map variables to groups.
    subset_cfg : dict
        Subset specification. Supports either:
        - {"subset": {"include_groups": [...], "exclude_groups": [...]}}, or
        - {"include_groups": [...], "exclude_groups": [...]}.

    Returns
    -------
    pandas.DataFrame
        Subset of X containing only selected columns that are present in X.

    Notes
    -----
    - Prints a warning listing how many variables were selected from `var_info`
      but not found in X (shows first 10).
    """
    subset = subset_cfg.get("subset", subset_cfg)  # supports either nesting style
    include_groups = subset.get("include_groups")
    exclude_groups = subset.get("exclude_groups")

    cols = columns_for_groups(var_info, include_groups, exclude_groups)

    # Keep only columns that actually exist in X
    cols_in_X = [c for c in cols if c in X.columns]

    # Helpful warnings (optional)
    missing = [c for c in cols if c not in X.columns]
    if missing:
        print(f"[subset] Warning: {len(missing)} columns in var_info not found in X (showing first 10): {missing[:10]}")

    return X.loc[:, cols_in_X]

def cfg_get(cfg, path):
    """
    Retrieve a nested configuration value using a list of keys.

    Walks through a nested dictionary `cfg` using the keys in `path`.
    Raises an informative KeyError if any key is missing.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary (possibly nested).
    path : list of str
        Sequence of keys specifying the nested lookup path.

    Returns
    -------
    object
        The value found at the specified path.

    Raises
    ------
    KeyError
        If any key in `path` is missing or an intermediate value is not a dict.
    """
    cur = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"Missing config key: {'.'.join(path)}")
        cur = cur[key]
    return cur

def make_run_dirs(cfg):
    """
    Create (if needed) and return the run results directory for a configuration.

    Uses cfg['run']['id'] to construct a path of the form:
    'Results/runs/<run_id>' and ensures it exists.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary expected to optionally contain {'run': {'id': ...}}.

    Returns
    -------
    pathlib.Path
        Path to the created/existing run directory.

    Notes
    -----
    - If cfg does not contain a run id, defaults to 'no_run_id'.
    """
    run_id = cfg.get("run", {}).get("id", "no_run_id")
    results_root = Path("Results") / "runs" / run_id
    results_root.mkdir(parents=True, exist_ok=True)
    return results_root

