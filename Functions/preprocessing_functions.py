import re
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from Functions.subsets import cfg_get

# import fixed params from base yaml:
with open("configs/base.yaml", "r") as f:
    cfg_base = yaml.safe_load(f) or {}
seed = cfg_get(cfg_base, ["params", "seed"])
random_state = seed

def log_drops(dropped_vars_reasons, vars_to_drop, reason):
    """
    Track dropped variables with reasons in a shared dict.

    Parameters
    ----------
    dropped_vars_reasons : dict
        Mapping of variable name -> set of reasons.
    vars_to_drop : list-like
        Variables being dropped.
    reason : str
        Reason for dropping.
    """
    for var in vars_to_drop:
        if var not in dropped_vars_reasons:
            dropped_vars_reasons[var] = set()
        dropped_vars_reasons[var].add(reason)

def get_redundant_pairs(df):
    """
    Return diagonal and lower-triangular column pairs for a correlation matrix.

    Useful for removing redundant correlation pairs when working with a
    symmetric correlation matrix (Var1-Var2 is the same as Var2-Var1) and
    excluding self-correlations.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame whose columns define the correlation matrix axes.

    Returns
    -------
    set of tuple
        Set of (col_i, col_j) pairs corresponding to the diagonal and the
        lower triangle (including diagonal).
    """
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, threshold, cor_m=None):
    """
    Compute absolute pairwise correlations and return pairs above a threshold.

    Computes absolute correlations between all column pairs (excluding redundant
    diagonal/lower triangle pairs) and returns both (a) pairs exceeding the
    threshold and (b) the full sorted list of non-redundant pairs.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing numeric features.
    threshold : float
        Absolute-correlation threshold. Pairs with abs(corr) >= threshold
        are returned in the filtered output.
    cor_m : pandas.DataFrame, optional
        Precomputed correlation matrix (same labels as `df.columns`). If provided,
        `df.corr(...)` is skipped and `cor_m` is used instead.

    Returns
    -------
    pandas.DataFrame
        DataFrame of correlations above threshold with columns:
        ['Var1', 'Var2', 'Corr'].
    pandas.DataFrame
        DataFrame of all non-redundant correlations sorted descending with columns:
        ['Var1', 'Var2', 'Corr'].

    Notes
    -----
    - If `cor_m` is None, correlations are computed with `df.corr(min_periods=50)`.
    - Redundant pairs (diagonal and lower triangle) are dropped.
    """
    if cor_m is None:
        au_corr = df.corr(min_periods=50).abs().unstack()
    else:
        au_corr = cor_m.abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    # Find common labels between the index and labels_to_drop
    common_labels = au_corr.index.intersection(labels_to_drop)
    # Drop common labels in one go
    au_corr = au_corr.drop(index=common_labels)
    # Sort the DataFrame after dropping
    au_corr = au_corr.sort_values(ascending=False)
    cors_above_threshold = au_corr[au_corr>=threshold]
    cors_above_threshold = pd.DataFrame(cors_above_threshold).reset_index()
    cors_above_threshold.columns = ["Var1", "Var2", "Corr"]
    all_cors = pd.DataFrame(au_corr).reset_index()
    all_cors.columns = ["Var1", "Var2", "Corr"]
    return cors_above_threshold, all_cors

def get_top_abs_phi(df, phi_coefs, threshold):
    """
    Return absolute Phi coefficient pairs above a threshold.

    Processes a Phi coefficient matrix similarly to correlation processing:
    flattens, drops redundant pairs, sorts by absolute strength, and filters
    by threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame whose columns define the variable names (used to generate
        redundant pairs to drop).
    phi_coefs : pandas.DataFrame
        Square matrix of Phi coefficients indexed/columned by variable names.
    threshold : float
        Absolute Phi threshold. Pairs with abs(phi) >= threshold are returned.

    Returns
    -------
    pandas.DataFrame
        Phi values above threshold with columns ['Var1', 'Var2', 'Phi'].
    pandas.DataFrame
        All non-redundant Phi values sorted descending with columns
        ['Var1', 'Var2', 'Phi'].
    """
    au_corr = phi_coefs.abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    # Find common labels between the index and labels_to_drop
    common_labels = au_corr.index.intersection(labels_to_drop)
    # Drop common labels in one go
    au_corr = au_corr.drop(index=common_labels)
    # Sort the DataFrame after dropping
    au_corr = au_corr.sort_values(ascending=False)
    phi_above_threshold = au_corr[au_corr>=threshold]
    phi_above_threshold = pd.DataFrame(phi_above_threshold).reset_index()
    phi_above_threshold.columns = ["Var1", "Var2", "Phi"]
    all_phi = pd.DataFrame(au_corr).reset_index()
    all_phi.columns = ["Var1", "Var2", "Phi"]
    return phi_above_threshold, all_phi

def drop_most_cor_var(cor_df):
    """
    Select variables to drop from a pairwise correlation table.

    Iterates through a correlation DataFrame and builds a list of variables
    to drop by repeatedly removing the variable that has the larger sum of
    correlations across remaining pairs.

    Parameters
    ----------
    cor_df : pandas.DataFrame
        DataFrame containing at least ['Var1', 'Var2', 'Corr'] where each row
        represents a variable pair and its correlation.

    Returns
    -------
    list
        List of variable names selected for dropping.

    Notes
    -----
    - If Corr == 1 for a pair, Var1 is added to the drop list (if not already),
      and all rows containing Var1 are removed from the search set.
    - This function modifies an internal copy (`search_df`) while iterating.
    """
    search_df = cor_df.copy()
    drop_list= []
    for index, row in search_df.iterrows():

        var_count_dict = {}
        var1 = row["Var1"]
        var2 = row["Var2"]

        # If correlation is 1, randomly delete 1 (it doesn't matter which as they are the same)
        if row['Corr'] == 1:

            # If not already in drop list, add var with highest corrs
            if var1 not in drop_list:
                drop_list.append(var1)

            # remove row in search df that has the higher value key in before continuing iteration
            search_df = search_df[
                ~search_df.apply(lambda row: row.astype(str).str.contains(var1)).any(axis=1)]

            continue

        running_sum_var1 = 0
        running_sum_var2 = 0

        # Loop through DataFrame rows
        for index, row in search_df.iterrows():
            if var1 in [row['Var1'], row['Var2']]:
                running_sum_var1 += row['Corr']
        for index, row in search_df.iterrows():
            if var2 in [row['Var1'], row['Var2']]:
                running_sum_var2 += row['Corr']

        var_count_dict[var1] = running_sum_var1
        var_count_dict[var2] = running_sum_var2

        # Find the key with the largest value
        key_with_higher_value = max(var_count_dict, key=var_count_dict.get)
        other_key = min(var_count_dict, key=var_count_dict.get)

        # If not already in drop list, add var with highest corrs
        if key_with_higher_value not in drop_list:
            drop_list.append(key_with_higher_value)

        # remove row in search df that has the higher value key in before continuing iteration
        search_df = search_df[~search_df.apply(lambda row: row.astype(str).str.contains(key_with_higher_value)).any(axis=1)]

    return drop_list



def remove_rows_by_values(df, values_to_remove):
    """
    Remove rows where Var1 or Var2 matches any value in a removal list.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns 'Var1' and 'Var2'.
    values_to_remove : list or set
        Variable names to filter out.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame excluding rows where 'Var1' or 'Var2' is in
        `values_to_remove`.
    """
    # Check if values in either column 1 or column 2 are in the list of values to remove
    mask = (df['Var1'].isin(values_to_remove)) | (df['Var2'].isin(values_to_remove))

    # Use boolean indexing to remove rows that match the condition
    df_filtered = df[~mask]

    return df_filtered



def drop_most_cor_variables_old(cor_df, keep_vars):
    """
    Heuristic selection of variables to drop from highly correlated pairs (legacy).

    Iterates through a correlation table and builds a set of variables to drop,
    attempting to respect a list of variables to keep.

    Parameters
    ----------
    cor_df : pandas.DataFrame
        Pairwise correlation table with columns ['Var1', 'Var2', 'Corr'].
    keep_vars : list or set
        Variables that should not be dropped if possible.

    Returns
    -------
    list
        List of variables selected for dropping.

    Notes
    -----
    - Prints intermediate row counts and may print a 'stop' message when no rows
      are removed in an iteration.
    - Uses a high-correlation shortcut (corr >= 0.99) to drop Var1 if allowed.
    - This function mutates an internal search DataFrame during iteration.
    """
    search_df = cor_df.copy()
    drop_set = set()

    # Iterate over the DataFrame in reverse order and remove rows
    for idx, row in search_df[::-1].iterrows():
        var1, var2, corr = row['Var1'], row['Var2'], row['Corr']
        n_rows = len(search_df)
        print(n_rows)
        var_count_dict = {}

        if len(search_df) > 0:
            # if corr >0.99, randomly drop Var 1
            if corr >= 0.99:
                if var1 not in keep_vars:
                    drop_set.add(var1)
                    search_df = search_df[~search_df[['Var1', 'Var2']].apply(lambda r: var1 in r.values, axis=1)]

            else:
                # else drop the var with the greater correlations with other vars
                for col in ['Var1', 'Var2']:
                    running_sum = search_df[search_df[['Var1', 'Var2']].apply(lambda r: var1 == r[col], axis=1)]['Corr'].sum()
                    var_count_dict[var1] = running_sum

                key_with_higher_value = max(var_count_dict, key=var_count_dict.get)
                if key_with_higher_value not in keep_vars:
                    drop_set.add(key_with_higher_value)
                    search_df = search_df[~search_df[['Var1', 'Var2']].apply(lambda r: key_with_higher_value in r.values,
                                                                             axis=1)]

        # search_df.reindex(inplace=True)
        end_n_rows = len(search_df)
        if n_rows == end_n_rows:
            print('stop:' + var1 + " + " +var2)

    return list(drop_set)



def drop_most_cor_variables_search(cor_df, keep_vars):
    """
    Select variables to drop from correlated pairs ***without modifying the iterator as it searches***.

    Builds a set of variables to drop based on correlation strength and each
    variable's total correlation mass across the full table, while attempting
    to respect keep_vars.

    Parameters
    ----------
    cor_df : pandas.DataFrame
        Pairwise correlation table with columns ['Var1', 'Var2', 'Corr'].
    keep_vars : list or set
        Variables that should be retained when possible.

    Returns
    -------
    list
        List of variables selected for dropping.

    Notes
    -----
    - If corr >= 0.99, Var1 is dropped unless it is in keep_vars.
    - Otherwise, the variable with the higher sum of correlations across
      all pairs is dropped (unless it is in keep_vars, then the other is dropped).
    - Prints progress for each row.
    """
    search_df = cor_df.copy()
    drop_set = set()

    # Iterate over the DataFrame
    for idx, row in search_df.iterrows():
        var1, var2, corr = row['Var1'], row['Var2'], row['Corr']
        var_count_dict = {}

        # if corr >0.99, randomly drop Var 1
        if corr >= 0.99:
            if var1 not in keep_vars:
                drop_set.add(var1)
        else:
            # else drop the var with the greater correlations with other vars

            # for var in [var1, var2]:
            #     if var == 'e53702y_D':
            #         print('stop')
            #     running_sum = search_df[search_df[['Var1', 'Var2']].apply(lambda r: var == r[var], axis=1)]['Corr'].sum()
            #     var_count_dict[var] = running_sum

            running_sum_var1 = 0
            running_sum_var2 = 0

            # Loop through DataFrame rows
            for index, row in search_df.iterrows():
                if var1 in [row['Var1'], row['Var2']]:
                    running_sum_var1 += row['Corr']
                if var2 in [row['Var1'], row['Var2']]:
                    running_sum_var2 += row['Corr']

            var_count_dict[var1] = running_sum_var1
            var_count_dict[var2] = running_sum_var2

            # Find the key with the larger value
            key_with_higher_value = max(var_count_dict, key=var_count_dict.get)
            key_with_lower_value = min(var_count_dict, key=var_count_dict.get)
            if key_with_higher_value not in keep_vars:
                drop_set.add(key_with_higher_value)
            else:
                drop_set.add(key_with_lower_value)

        print(f"{idx}/{len(search_df)}")

    return list(drop_set)



def remove_highly_correlated_columns(df, threshold, keep_vars):
    """
    Iteratively remove columns with correlations above a threshold.

    First removes (randomly) one variable from each pair with correlation >= 0.99.
    Then repeatedly identifies variables involved in correlations above `threshold`
    and drops the most collinear variable (by sum of absolute correlations),
    attempting to keep variables listed in `keep_vars`.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame (assumed numeric) to filter.
    threshold : float
        Correlation threshold for iterative pruning (e.g. 0.8).
    keep_vars : list or set
        Variables that should not be removed if possible.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame after removing correlated columns.
    set
        Set of all removed column names.

    Notes
    -----
    - Uses `np.corrcoef` in the initial >= 0.99 pass (can yield NaNs if columns
      contain NaNs or constant values).
    - The initial removal step is stochastic due to `np.random.choice`.
    - Prints progress messages and drop counts.
    - Modifies `df` in-place during the iterative loop after the initial drop.
    """
    all_columns_removed = set()

    # first, where correlation is >=0.99, randomly remove one (as they are essentially the same)
    correlated_variables = []

    for col1 in df.columns:
        for col2 in df.columns:
            if col1 != col2 and np.corrcoef(df[col1], df[col2])[0, 1] >= 0.99:
                correlated_variables.append((col1, col2))

    # Randomly select one variable to remove from each pair
    variables_to_remove = [np.random.choice(pair) for pair in correlated_variables]

    df = df.drop(variables_to_remove, axis=1)
    all_columns_removed.update(set(variables_to_remove))

    while True:
        # Calculate the absolute correlation matrix
        corr_m_abs = df.corr().abs()

        # Create a mask of 1s for upper triangle only
        mask = np.triu(np.ones(corr_m_abs.shape), k=1).astype(bool)

        # Extract upper triangular part of the correlation matrix
        corr_upper_tri = corr_m_abs.where(mask)

        # Find features which have at least one correlation above threshold
        highly_correlated_indices = (corr_upper_tri > threshold).any()

        # Filter out the indices that are not in keep_vars
        vars_in_keep_list = 0
        for var in keep_vars:
            if (var in highly_correlated_indices.index) and (highly_correlated_indices[var] == True):
                highly_correlated_indices[var] = False
                vars_in_keep_list =+1

        # Check if highly_correlated_indices is empty
        if len(highly_correlated_indices.value_counts()) == 2:
            counts_left = highly_correlated_indices.value_counts()[True]
        else:
            counts_left = 0
        print(f"No. correlated variables above {threshold} not in keep list: "
              f"{counts_left}, vars in keep list: {vars_in_keep_list}")
        if all(not item for item in highly_correlated_indices):
            break

        # Identify columns with the higher amount of collinearity with all other variables
        columns_to_remove = set()

        # iterate through the vars with at least one corr above threshold
        for index, col in enumerate(highly_correlated_indices[highly_correlated_indices].index):
            # for each col, find the cors with other variables above threshold, add to set
            correlated_cols = set(corr_m_abs[col][corr_m_abs[col] > threshold].index.tolist())
            correlated_cols.add(col)  # Check col to the set, add if not
            # find var with the highest correlations with other vars across full corr matrix
            most_cor_var = max(correlated_cols, key=lambda x: corr_m_abs[x].sum())
            # if most_cor_var not in keep vars, remove it
            if most_cor_var not in keep_vars:
                columns_to_remove.add(most_cor_var)
            # if it is, find next most correlated var to remove
            else:
                # remove index from correlated columns, and find next most cor var to remove
                correlated_cols.remove(most_cor_var)
                next_most_cor_var = max(correlated_cols, key=lambda x: corr_m_abs[x].sum())
                # check not in keep_vars
                if next_most_cor_var not in keep_vars:
                    # if not, remove it
                    columns_to_remove.add(most_cor_var)
                else:
                    # print to see if any cases where top two most corr vars also in keep_vars
                    print(f'keeping vars (correlated {round(corr_m_abs[most_cor_var][next_most_cor_var], 2)})'
                          f'but in keep_list): {most_cor_var} + {next_most_cor_var}')

        # Remove the identified columns from the DataFrame
        drop_list = list(columns_to_remove)
        df.drop(columns=drop_list, inplace=True)
        print(f"Dropping {len(drop_list)}, no. of columns: {df.shape[1]}")
        # update total list of columns removed
        all_columns_removed.update(set(columns_to_remove))

    return df, all_columns_removed



def check_condition(value):
    """
    Normalize underscore-suffixed strings by stripping qualifying suffixes.

    If a string has the form '<prefix>_<suffix>' and the suffix is either:
    - 'applicable'
    - an integer string (e.g. '3')
    - a number with up to 1 decimal place (e.g. '2.5')
    then returns '<prefix>'. Otherwise returns the original value.

    Parameters
    ----------
    value : str
        Input string to check.

    Returns
    -------
    str
        Prefix with suffix removed if the suffix matches the rule, otherwise
        the original string.
    """
    parts = value.rsplit('_', 1)
    if len(parts) == 2:
        after_underscore = parts[1]
        if after_underscore == 'applicable' or after_underscore.isdigit() or\
                (after_underscore.replace('.', '', 1).isdigit() and len(after_underscore.split('.')[-1]) <= 1):
            return parts[0]
    return value


def flatten_dict(d, sep='_'):
    """
    Flatten a nested dictionary by recursively removing intermediate keys.

    Recursively traverses nested dictionaries and returns a single-level
    dictionary containing only the leaf keys and values.

    Parameters
    ----------
    d : dict
        Input dictionary (possibly nested).
    sep : str, default='_'
        Separator between keys (currently not used in this implementation).

    Returns
    -------
    dict
        Flattened dictionary containing leaf keys mapped to leaf values.

    Notes
    -----
    - This implementation drops parent key context rather than composing
      hierarchical keys. If nested dicts contain duplicate leaf keys, later
      keys will overwrite earlier ones.
    """
    items = []
    for k, v in d.items():
        if isinstance(v, dict):
            # If the value is a dictionary, flatten it recursively
            items.extend(flatten_dict(v, sep=sep).items())
        else:
            items.append((k, v))
    return dict(items)



def drop_cols(col_list, df):
    """
    Drop a list of columns from a DataFrame and return updated DataFrame and columns.

    Parameters
    ----------
    col_list : list
        Column names to drop (if present).
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame
        Copy of `df` with columns removed.
    list
        List of remaining column names after dropping.
    """
    dropped_df = df.copy()
    dropped_col_list = list(df.columns)
    for col in col_list:
        if col in df.columns:
            dropped_df.drop(col, inplace=True, axis=1)
            dropped_col_list.remove(col)
    return dropped_df, dropped_col_list




def plot_category_distribution(df, save_path, min_cat, categorical_cols, var_name_dict):
    """
    Plot bar charts for categorical variables with small minimum category counts.

    For each categorical column, computes value counts and saves a bar plot if
    the smallest category count is <= `min_cat`.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    save_path : str
        Directory path where figures will be saved (string concatenation is used).
    min_cat : int
        Minimum-count threshold triggering a plot.
    categorical_cols : list
        List of categorical column names to check.
    var_name_dict : dict
        Mapping from column name to human-readable long name used in the title.

    Returns
    -------
    None

    Notes
    -----
    - Saves one PNG per qualifying categorical column using filename f'{col}.png'.
    - Writes counts above bars.
    """
    for col in categorical_cols:
        long_name = var_name_dict[col]
        if col in df.columns:
            category_counts = df[col].value_counts()
            if category_counts.min() <= min_cat:
                plt.figure(figsize=(8, 6))
                category_counts.plot(kind='bar')
                plt.title(f'Distribution of {col}: \n{long_name}')
                plt.xlabel(col)
                plt.ylabel('Count')
                # Adding count values above the bars
                for i, count in enumerate(category_counts):
                    plt.text(i, count + 0.5, str(count), ha='center', va='bottom')
                plt.savefig(save_path + f'{col}.png')
                plt.close()
    return



def count_categories_to_file(df, output_file, categorical_columns, var_name_dict, min_cat, cat_name_dict, data_name):
    """
    Write category counts for selected categorical variables to a text file.

    For each categorical variable, if its minimum category count is <= `min_cat`,
    writes the variable name, long name, and per-category counts (optionally
    including category labels from `cat_name_dict`) to `output_file`.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    output_file : str
        Path to the output text file to write.
    categorical_columns : list
        List of categorical column names to process.
    var_name_dict : dict
        Mapping from variable name to long descriptive name.
    min_cat : int
        Minimum-count threshold for including a variable in the report.
    cat_name_dict : dict
        Nested mapping used to look up category labels by variable and category code.
        Expected shape: {<outer_key>: {<var>: {<category_code_str>: <label>}}}
    data_name : str
        Name of the dataset (used in printed progress messages).

    Returns
    -------
    int
        Number of variables written to file.
    list
        List of variable names that met the threshold criterion.

    Notes
    -----
    - Prints progress messages for included variables.
    - Categories are cast to int then to str before lookup.
    - Only writes category lines when a label is successfully found in cat_name_dict.
    """
    with open(output_file, 'w') as file:
        file.write(f"Category counts for categorical variables where minimum category is <= {min_cat}\n")
        var_count = 1
        list = []
        for column in categorical_columns:
            if column in df.columns:
                category_counts = df[column].value_counts().sort_values(ascending=False)
                if (category_counts.min() <= min_cat) == True:
                    # todo: neet to make sure works excluding -999
                    print(f"{data_name} -- column with less than {str(min_cat)} category counts: {column} -- {var_count}")
                    print(f"{data_name} -- category counts for {column}:")
                    list.append(column)
                    long_name = var_name_dict[column]
                    file.write(f"{var_count}. {column}: {long_name} ...\n")
                    for category, count in category_counts.items():
                        category = str(int(category))
                        cat_name_bool = False
                        for key, outer_dict in cat_name_dict.items():
                            try:
                                cat_name = outer_dict[column][category]
                                cat_name = cat_name.strip()
                            except KeyError:
                                cat_name_bool = False
                                cat_name = "no name"
                            else:
                                cat_name_bool = True
                            finally:
                                if cat_name_bool == True:
                                    file.write(f"Category: {category} ({cat_name}): {count}\n")
                                    print(f"  Category: {category} ({cat_name}): {count}")
                                else:
                                    file.write(f"Category: {category}: {count}\n")
                                    print(f"  Category: {category}: {count}")
                    file.write("\n")
                    var_count += 1
    return var_count-1, list


def find_matching_rows(column_data, search_list):
    """
    Find indices of rows containing parenthesized substrings matching a search list.

    Extracts all substrings inside parentheses for each value in `column_data`
    and returns the indices where any extracted substring is present in `search_list`.

    Parameters
    ----------
    column_data : iterable of str
        Sequence of strings to search.
    search_list : list or set
        Set of target substrings to match.

    Returns
    -------
    list
        List of integer indices of matching rows.
    """
    matching_rows = []
    for index, value in enumerate(column_data):
        # Find all substrings within parentheses
        substrings = re.findall(r'\((.*?)\)', value)
        # Check if any substring matches the search list
        if any(substring in search_list for substring in substrings):
            matching_rows.append(index)
    return matching_rows


def remove_variables(df, variables_to_remove):
    """
    Remove columns from a DataFrame in-place if they exist.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame to modify.
    variables_to_remove : list
        Column names to drop if present.

    Returns
    -------
    pandas.DataFrame
        The modified DataFrame (same object as input).

    Notes
    -----
    - Drops columns in-place (`inplace=True`).
    - Prints a message for each removed variable and prints the final shape.
    """
    # Remove variables if they exist in the DataFrame
    for var in variables_to_remove:
        if var in df.columns:
            df.drop(columns=var, inplace=True)
            print(f"Variable '{var}' removed from DataFrame.")
    print(f"new dataframe shape: {df.shape}")
    return df