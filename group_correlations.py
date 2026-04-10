import os
import re
import json
import pandas as pd
import numpy as np

cor_save_path = "Outputs/IV_correlations/Grouped/"

if __name__ == "__main__":
    X_test_p = pd.read_csv("Data/Preprocessed/X_train_p.csv")
    X_and_y = pd.read_csv("Data/Preprocessed/X_and_y.csv")
    with open('Data/Meta/extended_group_mapping.json', 'r') as f:
        block_dict = json.load(f)

    # Map columns to groups
    group_to_columns = {}
    for col, group in block_dict.items():
        group_to_columns.setdefault(group, []).append(col)

    # reorder df cols
    block_vars = []
    block_names = set()
    double_block_var_names = []
    for block_name, all_block_vars in group_to_columns.items():
        for var in all_block_vars:
            if var in X_test_p.columns.to_list():
                block_vars.append(var)
                block_names.add(block_name)
                double_block_var_names.append(block_name + "_" + var)

    df_ordered = X_test_p[block_vars]

    # create cor matrix
    cor = round(df_ordered.corr(), 2)
    cor.columns = double_block_var_names
    cor.index = double_block_var_names
    cor.to_excel(cor_save_path+"Grouped_correlation_matrix2.xlsx")

    # ----------------------------------------
    # find cors above 0.7 in different groups:

    # Extract group names from column/index names
    def extract_group(name):
        return name.split('_')[0]

    threshold = 0.6
    save_file = "Outputs/IV_correlations/Grouped/" + f"Cors_different_groups_above{threshold}.txt"
    # Find correlations > 0.7 between variables from different groups
    with open(save_file, "w") as f:
        print(f"Correlations above {threshold} in different groups:", file=f)

        for i in range(len(cor.columns)):
            for j in range(i+1, len(cor.columns)):
                col1, col2 = cor.columns[i], cor.columns[j]
                if (extract_group(col1) != extract_group(col2) and
                    cor.iloc[i, j] > threshold):
                    print(f"---------> different group: {col1} - {col2}: {cor.iloc[i, j]:.2f}")
                    print(f"{col1} - {col2}: {cor.iloc[i, j]:.2f}", file=f)

                elif (cor.iloc[i, j] > threshold):
                    print(f"high cor, same group: {col1} - {col2}: {cor.iloc[i, j]:.2f}")



    # # plot heatmap
    # lines = []
    # line_int = 0
    # for block, vars_list in new_block_dict.items():
    #     block_len = len(vars_list)
    #     line_int = line_int + block_len
    #     lines.append(line_int)
    #
    # label_geo_list=[]
    # line_int = 0
    # for block, vars_list in new_block_dict.items():
    #     block_len = len(vars_list)
    #     if block_len > 1:
    #         midpoint = block_len/2
    #     else:
    #         midpoint = 1
    #     line_geo = line_int + midpoint
    #     label_geo_list.append(line_geo)
    #     line_int = line_int + block_len
    #
    # group_labels = ["Prior Ach.", "Grades", "School Track", "Demo. & SES",
    #                 "IQ", "Motiv. & Emot.", "Fam. Context (S,P)",
    #                 "Cog. Strat.", "Class Context (S)", "Class Context (T)"]
    #
    print('done!')