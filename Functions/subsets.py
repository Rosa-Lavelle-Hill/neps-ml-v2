import pandas as pd
from pathlib import Path 

def load_var_info(csv_path: str) -> pd.DataFrame:
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
    group_col: str = "Variable_Group",
):
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
        cur = cfg
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                raise KeyError(f"Missing config key: {'.'.join(path)}")
            cur = cur[key]
        return cur

def make_run_dirs(cfg):
    run_id = cfg.get("run", {}).get("id", "no_run_id")
    results_root = Path("Results") / "runs" / run_id
    results_root.mkdir(parents=True, exist_ok=True)
    return results_root

