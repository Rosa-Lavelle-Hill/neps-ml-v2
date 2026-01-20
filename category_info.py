import pandas as pd
import pyreadstat
import os
import json

directory_path = "Data/Raw/"

# List all files in the directory
file_list = os.listdir(directory_path)

data_dict = {}
meta_dict = {}

# Iterate through each file in the directory
for filename in file_list:
    # Full path to the file
    file_path = os.path.join(directory_path, filename)

    short_name = filename.split("_", 2)[1]

    # Check if the path is a regular file (not a directory)
    if os.path.isfile(file_path):

        # Process the file
        # df, meta = pyreadstat.read_dta(file_path, encoding="ISO-8859-1", metadataonly=True)
        df, meta = pyreadstat.read_dta(file_path, encoding="latin1", metadataonly=True)

        # Store data
        data_dict[short_name] = df
        meta_dict[short_name] = meta

var_cat_labels_dict = {}

for data_name, meta_data in meta_dict.items():
    var_cat_labels_dict[data_name] = meta_data.variable_value_labels

with open('Data/Meta/label_info_nested_dict.json', 'w') as file:
    json.dump(var_cat_labels_dict, file, indent=4)

if __name__ == "__main__":

    labels_df = pd.DataFrame.from_dict(var_cat_labels_dict, orient='columns')
    nan_count_per_row = labels_df.isna().sum(axis=1)
    labels_df = labels_df[1:].stack().reset_index()
    labels_df.columns = ['Variable', "Dataset", "Labels"]
    labels_df.to_csv("Data/Meta/label_info_all.csv")
    print(labels_df.shape)

    # Select only vars in our dataset
    df = pd.read_csv("Data/Preprocessed/df_R_processed.csv", low_memory=False, index_col=[0])
    var_list = df.columns
    subset_df = labels_df[labels_df['Variable'].isin(var_list)]
    subset_df.to_csv("Data/Meta/label_info.csv")
    print(subset_df.shape)

    # Select only vars with cat zero in (to inspect)
    zero_vars = pd.read_csv("Outputs-Linear/category_distributions/zero_cat_vars.csv")
    zero_vars = list(zero_vars.iloc[:, 0])
    zero_subset_df = subset_df[subset_df['Variable'].isin(zero_vars)]
    print(zero_subset_df.shape)

    # Count the number of zeros in each column
    df_zero_counts = df[zero_vars].applymap(lambda x: str(x))
    zero_counts = df_zero_counts.apply(lambda x: x.eq('0').sum())
    zero_counts = zero_counts.reset_index()
    zero_counts.columns = ["Variable", "Zero_counts"]
    zero_counts_var_list = list(zero_counts[zero_counts["Zero_counts"]>0]['Variable'])
    df_zero_counts = zero_subset_df[zero_subset_df['Variable'].isin(zero_counts_var_list)]
    df_zero_counts.to_csv("Data/Meta/zero_vars_label_info.csv")
    print(df_zero_counts.shape)
    print('done')

# todo: some variables don't have a zero as a category -> count zero's for each col