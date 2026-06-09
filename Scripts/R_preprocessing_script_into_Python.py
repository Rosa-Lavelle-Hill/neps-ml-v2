import os
import re
import numpy as np
import pandas as pd

def main():
    print("Running pre pre-processing script (formerly R code)!")

    # ----------------------------
    # Functions
    # ----------------------------
    def safe_to_numeric(series: pd.Series) -> pd.Series:
        """Try to coerce to numeric; non-coercible values become NaN."""
        return pd.to_numeric(series, errors="coerce")

    def cronbach_alpha(df_items: pd.DataFrame) -> float:
        """
        Cronbach's alpha for item columns (expects numeric, with NaNs allowed).
        Uses pairwise deletion by dropping rows with any NaN across items.
        """
        x = df_items.copy()
        x = x.dropna(axis=0, how="any")
        k = x.shape[1]
        if k < 2 or x.shape[0] < 2:
            return np.nan

        item_vars = x.var(axis=0, ddof=1)
        total_var = x.sum(axis=1).var(ddof=1)
        if total_var == 0 or np.isnan(total_var):
            return np.nan
        return float((k / (k - 1)) * (1 - (item_vars.sum() / total_var)))

    def print_drop(before_rows, before_cols, after_rows, after_cols, why: str):
        print(
            f"{why} | rows removed: {before_rows - after_rows}, "
            f"variables removed: {before_cols - after_cols}"
        )

    # ----------------------------
    # Paths / IO
    # ----------------------------
    data_path = "Data/neps.csv"
    varinfo_path = "Data/Meta/variable_info_agreed_Feb_2_2026.csv"

    out_missing_pct = "Outputs/missing/R_mising_-100_perc.csv"
    out_alpha = "Outputs/R_preprocessing/alpha_for_aggregated_vars.csv"
    out_df = "Data/Preprocessed/df_R_to_python_processed.csv"

    os.makedirs(os.path.dirname(out_missing_pct), exist_ok=True)
    os.makedirs(os.path.dirname(out_alpha), exist_ok=True)
    os.makedirs(os.path.dirname(out_df), exist_ok=True)

    # ----------------------------
    # Read data
    # ----------------------------
    df = pd.read_csv(data_path)
    variable_info = pd.read_csv(varinfo_path)

    print(f"Loaded df: {df.shape[0]} rows, {df.shape[1]} variables")
    print(f"Loaded variable_info: {variable_info.shape[0]} rows, {variable_info.shape[1]} columns")

    # ----------------------------
    # Subset to agreed variables
    # ----------------------------
    before = df.shape
    mask = (variable_info["include as predictor"] == 1) | (variable_info["include as ID"] == 1)
    variables = variable_info.loc[mask, "var"].astype(str)
    variables = variables[variables.notna()].tolist()

    # Keep only columns that actually exist in df (and report missing)
    existing = [v for v in variables if v in df.columns]
    missing = [v for v in variables if v not in df.columns]

    df = df.loc[:, existing]
    after = df.shape
    print_drop(before[0], before[1], after[0], after[1], "Subset to agreed variables")
    if missing:
        print(f"NOTE: {len(missing)} agreed variables not found in df and therefore could not be selected:")
        print("Missing columns:", missing)

    # ----------------------------
    # Check that all variables are numeric-ish
    # (R version: as_numeric_trycatch + print 'check var' on error/warning)
    # Here: flag columns that contain non-numeric strings (excluding NA).
    # ----------------------------
    check_cols = []
    for col in df.columns:
        s = df[col]
        # Work on strings, but ignore NA-like
        s_str = s.astype("string")
        s_num = pd.to_numeric(s_str, errors="coerce")
        # If there are non-missing entries that fail numeric coercion -> "check"
        bad = s_str.notna() & s_num.isna()
        if bad.any():
            check_cols.append(col)

    print(f"Numeric coercion check: {len(check_cols)} columns contain non-numeric entries that would coerce to NaN.")
    for c in check_cols[:50]:
        print(f"check {c}")
    if len(check_cols) > 50:
        print(f"... (showing first 50 of {len(check_cols)})")

    # ----------------------------
    # Count specific values (-54, 93, -99) as percentages per column
    # (R counted as strings in df_str)
    # ----------------------------
    values_to_count = ["-54", "93", "-99"]
    df_str = df.astype("string")

    percentages = pd.DataFrame(index=df.columns, columns=values_to_count, dtype=float)
    for col in df_str.columns:
        col_data = df_str[col]
        total_data_points = col_data.notna().sum()
        for v in values_to_count:
            if total_data_points == 0:
                percentages.loc[col, v] = np.nan
            else:
                count = (col_data == v).sum(skipna=True)
                percentages.loc[col, v] = round(count / total_data_points * 100, 2)

    # R script did rowMeans across the matrix; this yields per-variable average across counted codes
    percentages["Average"] = percentages.mean(axis=1, skipna=True)
    # And it appended a final "Average" row; in R this was actually rowMeans(matrix) then rbind,
    # but conceptually we'll provide an overall average across variables too:
    overall = pd.DataFrame(
        [percentages[values_to_count + ["Average"]].mean(axis=0, skipna=True)],
        index=["Average"],
    )
    percentages_out = pd.concat([percentages[values_to_count + ["Average"]], overall], axis=0)

    percentages_out.to_csv(out_missing_pct)
    print(f"Saved missing-code percentages to: {out_missing_pct}")

    # ----------------------------
    # Recode missingness
    # -93 or NA -> -100 (NOT APPLICABLE / by design)
    # -99, -54, -94, -95, -97, -98, -90, -91, -92, -56, -29..-20 -> -200 (RANDOM)
    # -52, -53, -55 -> -300 (EDITORIAL)
    # ----------------------------
    before_rows, before_cols = df.shape

    # Coerce to numeric for recoding, but preserve non-numeric as-is (they'll become NaN)
    df_num = df.apply(safe_to_numeric)

    # Track how many cells are changed for each recode block
    changed_100 = 0
    changed_200 = 0
    changed_300 = 0

    # -100
    mask_100 = df_num.isna() | df_num.isin([-93])
    changed_100 = int(mask_100.sum().sum())
    df_num = df_num.mask(mask_100, -100)

    # -200
    vals_200 = [-99, -54, -94, -95, -97, -98, -90, -91, -92, -56] + list(range(-29, -19))
    mask_200 = df_num.isin(vals_200)
    changed_200 = int(mask_200.sum().sum())
    df_num = df_num.mask(mask_200, -200)

    # -300
    mask_300 = df_num.isin([-52, -53, -55])
    changed_300 = int(mask_300.sum().sum())
    df_num = df_num.mask(mask_300, -300)

    df = df_num  # continue with recoded numeric frame

    after_rows, after_cols = df.shape
    print_drop(before_rows, before_cols, after_rows, after_cols, "Recode missingness (no row/variable removal expected)")
    print(f"Cells recoded to -100 (NA/-93): {changed_100}")
    print(f"Cells recoded to -200 (random missing group): {changed_200}")
    print(f"Cells recoded to -300 (editorial missing group): {changed_300}")

    # ----------------------------
    # Date merge: month+year to date string (YYYY-MM-DD)
    # R logic:
    #   monthvar = vars where date_merge == "y" and var name ends with "m"
    #   recode month categories: 21->1, 24->4, 27->7, 30->10, 32->12
    #   for each monthvar i:
    #       newname = remove trailing "m"
    #       yearvar = newname + "y"
    #       df[newname] = df[i] (copy)
    #       for rows where month in 1..12: df[newname] = as.Date("01/mm/yyyy")
    #       drop yearvar and i
    # ----------------------------
    before = df.shape

    # Ensure columns exist and variable_info has needed columns
    monthvar = []
    if "date_merge" in variable_info.columns:
        candidates = variable_info.loc[
            (variable_info["date_merge"] == "y") & variable_info["var"].astype(str).str.contains(r"m\b", regex=True),
            "var",
        ].astype(str).tolist()
        # Keep only those actually present
        monthvar = [c for c in candidates if c in df.columns]

    # Recode less specific month categories
    month_map = {21: 1, 24: 4, 27: 7, 30: 10, 32: 12}
    month_recode_cells = 0
    for mcol in monthvar:
        before_vals = df[mcol].copy()
        df[mcol] = df[mcol].replace(month_map)
        month_recode_cells += int((before_vals != df[mcol]).sum())

    print(f"Date-merge month recode: changed {month_recode_cells} cells across {len(monthvar)} month columns.")

    # Merge month/year -> date string
    dropped_date_cols = 0
    created_date_cols = 0
    for mcol in monthvar:
        newname = re.sub(r"m\b", "", mcol)  # remove trailing 'm'
        yearvar = f"{newname}y"

        # If the corresponding year column doesn't exist, skip 
        if yearvar not in df.columns:
            print(f"NOTE: skipping date merge for {mcol} because year column {yearvar} not found.")
            continue

        # Copy to preserve missingness codes (will become object after mixing with date strings)
        df[newname] = df[mcol].copy()
        created_date_cols += 1

        # Create date strings only for actual months 1..12 and non-missing year
        is_real_month = df[newname].isin(list(range(1, 13)))
        is_real_year = df[yearvar].notna() & (~df[yearvar].isin([-100, -200, -300, -400]))
        ok = is_real_month & is_real_year

        # Convert year to int safely
        years = pd.to_numeric(df.loc[ok, yearvar], errors="coerce").astype("Int64")
        months = pd.to_numeric(df.loc[ok, mcol], errors="coerce").astype("Int64")

        dates = pd.to_datetime(
            dict(year=years.astype(int), month=months.astype(int), day=1),
            errors="coerce",
        ).dt.date.astype(str)

        # Convert target col to object, then assign strings
        df[newname] = df[newname].astype("object")
        df.loc[ok, newname] = dates.values

        # Drop original cols
        df.drop(columns=[yearvar, mcol], inplace=True, errors="ignore")
        dropped_date_cols += 2
        

    after = df.shape
    print_drop(before[0], before[1], after[0], after[1], "Merge month/year date variables (drops original m/y columns)")
    print(f"Created date columns: {created_date_cols}; dropped m/y columns: {dropped_date_cols}")
    print("df shape: ", df.shape)

    # ----------------------------
    # Aggregate variables (top-down decision)
    # - For each aggregate name:
    #   * take items with that aggregate
    #   * create no-miss version where -100/-200/-300 -> NA
    #   * reverse code if needed: (reverse_scale_max+1) - value
    #   * agg = row mean (skip NA)
    #   * if agg is NA (all items missing):
    #       - if all original item values in that row are identical -> set agg to that missing code
    #       - else -> set agg to -400
    #   * add df[aggregate] = agg
    #   * compute alpha on no-miss items
    #   * drop original items from df
    # ----------------------------
    if "aggregate" in variable_info.columns:
        agg_names = (
            variable_info.loc[variable_info["aggregate"].astype(str) != "", "aggregate"]
            .astype(str)
            .unique()
            .tolist()
        )
        agg_names = [name for name in agg_names if name != "nan"]  # in case empty strings were read as "nan"
    else:
        agg_names = []

    aggregated_vars = pd.DataFrame({"var": agg_names, "alpha": np.nan})

    total_removed_vars_from_agg = 0

    for idx, agg_name in enumerate(agg_names, start=1):
        item_vars = (
            variable_info.loc[variable_info["aggregate"].astype(str) == agg_name, "var"]
            .astype(str)
            .tolist()
        )
        item_vars = [v for v in item_vars if v in df.columns]
        if len(item_vars) == 0:
            print(f"{agg_name}: no matching item variables found in df; skipping.")
            continue

        before = df.shape

        for_agg = df[item_vars].copy()

        # no-miss version
        for_agg_no_miss = for_agg.copy()
        for_agg_no_miss = for_agg_no_miss.replace({-100: np.nan, -200: np.nan, -300: np.nan})

        # reverse code where required
        if "reverse" in variable_info.columns:
            for v in item_vars:
                row = variable_info.loc[variable_info["var"].astype(str) == v]
                rev_val = row["reverse"].iloc[0] if ("reverse" in row.columns and not row.empty) else 0

                # Treat NaN / blanks as 0 (no reverse)
                if pd.isna(rev_val):
                    rev_flag = 0
                else:
                    # handle strings like "1", "0"
                    try:
                        rev_flag = int(float(rev_val))
                    except Exception:
                        print(f"NOTE: reverse flag for {v} is non-numeric ({rev_val}); treating as 0.")
                        rev_flag = 0

                if rev_flag == 1:
                    if "reverse_scale_max" not in row.columns or pd.isna(row["reverse_scale_max"].iloc[0]):
                        print(f"NOTE: reverse requested for {v} but reverse_scale_max missing/NaN; skipping reverse.")
                    else:
                        rmax = float(row["reverse_scale_max"].iloc[0])
                        for_agg_no_miss[v] = (rmax + 1) - for_agg_no_miss[v]

        # aggregate: mean across items ignoring NaN
        agg = for_agg_no_miss.mean(axis=1, skipna=True)

        # reintegrate missingness for rows where mean is NaN
        na_rows = agg.isna()
        if na_rows.any():
            # If all original item values in a row are identical -> carry that missing code
            # else -> -400
            orig = for_agg.loc[na_rows, item_vars]
            # nunique across columns (ignoring NaNs shouldn't matter here because codes are numeric;
            # but we’ll be explicit)
            nunique = orig.apply(lambda r: pd.Series(r.values).dropna().nunique(), axis=1)
            # If a row has all NaNs (unlikely after recode, but possible), treat as mixed -> -400
            same_code = (nunique == 1)
            # Pull the first item as the code to copy (same across items by definition)
            first_item = orig[item_vars[0]]
            agg.loc[na_rows & same_code] = first_item.loc[same_code].values
            agg.loc[na_rows & (~same_code)] = -400

        # add aggregate to df
        df[agg_name] = agg

        # alpha on no-miss items
        alpha_val = cronbach_alpha(for_agg_no_miss[item_vars])
        aggregated_vars.loc[aggregated_vars["var"] == agg_name, "alpha"] = alpha_val

        # drop original item vars
        df.drop(columns=item_vars, inplace=True, errors="ignore")
        removed_now = len(item_vars)
        total_removed_vars_from_agg += removed_now

        after = df.shape
        print_drop(before[0], before[1], after[0], after[1], f"Aggregate '{agg_name}' (dropped its item variables)")
        print(f"{agg_name}: {idx}/{len(agg_names)} | items: {removed_now} | alpha: {alpha_val}")

    print(f"Total variables removed due to aggregation: {total_removed_vars_from_agg}")
    print(f"df shape: {df.shape}")
    # ----------------------------
    # Save outputs
    # ----------------------------
    aggregated_vars.to_csv(out_alpha, index=False)
    print(f"Saved alpha for aggregated vars to: {out_alpha}")

    df.to_csv(out_df, index=False)
    print(f"Saved processed df to: {out_df}")

    print(f"Final df dimensions: {df.shape[0]} rows, {df.shape[1]} variables")
    print("finished running Pre-pre-Python preprocessing script (formerly R code)!")
    #===============================================================================
    # check preprocessed df is the same:

    df_py = df.copy()
    df_r = pd.read_csv("Data/Preprocessed/df_R_processed.csv", index_col=[0])

    print(f"Rows: python ={df_py.shape[0]}, R ={df_r.shape[0]}")
    print(f"Columns: python ={df_py.shape[1]}, R ={df_r.shape[1]}")

    set_py = set(df_py.columns)
    set_r = set(df_r.columns)

    print("Only in Python:", set_py - set_r)
    print("Only in R:", set_r - set_py)

    print("done with check!")
    #===============================================================================



if __name__ == "__main__":
    main()

