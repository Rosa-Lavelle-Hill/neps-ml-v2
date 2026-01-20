import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from fixed_params import seed, imputer_max_iter

random_state = seed

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, threshold, cor_m=None):
    '''
    Computes pairwise correlation of columns, excluding NA/null values.
    :param df: dataframe containing features to compute correlations
    :param threshold: threshold for correlation above which to return
    :param cor_m: correlation matrix, optional instead of df
    :return: dataframe containing pairwise correlations above threshold, all pairwise correlations
    '''
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
    # Check if values in either column 1 or column 2 are in the list of values to remove
    mask = (df['Var1'].isin(values_to_remove)) | (df['Var2'].isin(values_to_remove))

    # Use boolean indexing to remove rows that match the condition
    df_filtered = df[~mask]

    return df_filtered



def drop_most_cor_variables_old(cor_df, keep_vars):

    search_df = cor_df.copy()
    drop_set = set()

    # drop_values = set(search_df[search_df["Corr"] == 1]['Var1'].unique())
    # # remove any variables from drop list that are in the "keeps" list
    # drop_values.difference_update(remove_vars)
    # # remove rows from search_df which have a variable in drop_values list
    # search_df = remove_rows_by_values(search_df, drop_values)
    # drop_set.update(drop_values)

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
    """Doesn't modify the search_df as it iterates"""

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
    """Function to check if a string is a number with 0 or 1 decimal places or is 'applicable'"""
    parts = value.rsplit('_', 1)
    if len(parts) == 2:
        after_underscore = parts[1]
        if after_underscore == 'applicable' or after_underscore.isdigit() or\
                (after_underscore.replace('.', '', 1).isdigit() and len(after_underscore.split('.')[-1]) <= 1):
            return parts[0]
    return value


def flatten_dict(d, sep='_'):
    """
    Flatten a nested dictionary by removing the top layer.

    Parameters:
    - d: The input dictionary.
    - sep: The separator to be used between keys.

    Returns:
    - A flattened dictionary.
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
    dropped_df = df.copy()
    dropped_col_list = list(df.columns)
    for col in col_list:
        if col in df.columns:
            dropped_df.drop(col, inplace=True, axis=1)
            dropped_col_list.remove(col)
    return dropped_df, dropped_col_list


def impute_X(X_train, X_test, num_cols, cat_cols):
    # Define imputers
    imp_iter_num = IterativeImputer(missing_values=np.nan,
                                    max_iter=imputer_max_iter,
                                    random_state=random_state)

    imp_iter_cat = IterativeImputer(estimator=RandomForestClassifier(),
                                    initial_strategy='most_frequent',
                                    missing_values=np.nan,
                                    max_iter=imputer_max_iter,
                                    random_state=random_state,
                                    )
    # Define two separate pipelines
    numerical_pipeline = Pipeline([
        ('imputer_num', imp_iter_num)
    ])
    categorical_pipeline = Pipeline([
        ('imputer_cat', imp_iter_cat)
    ])
    # Create a column transformer
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, num_cols),
        ('cat', categorical_pipeline, cat_cols)
    ])
    # Fit and transform data
    X_train_imputed = preprocessor.fit_transform(X_train)
    X_test_imputed = preprocessor.transform(X_test)

    # Concat num and cat data
    transformed_num_data_train = pd.DataFrame(X_train_imputed[:, :len(num_cols)], columns=num_cols)
    transformed_cat_data_train = pd.DataFrame(X_train_imputed[:, len(num_cols):], columns=cat_cols)
    X_train_imputed = pd.concat([transformed_num_data_train, transformed_cat_data_train], axis=1)

    transformed_num_data_test = pd.DataFrame(X_test_imputed[:, :len(num_cols)], columns=num_cols)
    transformed_cat_data_test = pd.DataFrame(X_test_imputed[:, len(num_cols):], columns=cat_cols)
    X_test_imputed = pd.concat([transformed_num_data_test, transformed_cat_data_test], axis=1)

    return X_train_imputed, X_test_imputed


def plot_category_distribution(df, save_path, min_cat, categorical_cols, var_name_dict):
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
                    file.write("\n")
                    var_count += 1


    return var_count-1, list


def find_matching_rows(column_data, search_list):
    matching_rows = []
    for index, value in enumerate(column_data):
        # Find all substrings within parentheses
        substrings = re.findall(r'\((.*?)\)', value)
        # Check if any substring matches the search list
        if any(substring in search_list for substring in substrings):
            matching_rows.append(index)
    return matching_rows


def remove_variables(df, variables_to_remove):
    # Remove variables if they exist in the DataFrame
    for var in variables_to_remove:
        if var in df.columns:
            df.drop(columns=var, inplace=True)
            print(f"Variable '{var}' removed from DataFrame.")
    print(f"new dataframe shape: {df.shape}")
    return df