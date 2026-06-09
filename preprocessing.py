# https://www.neps-data.de/Portals/0/NEPS/Datenzentrum/Forschungsdaten/SC6/13-0-0/SC6_13-0-0_DataManual.pdf
# python preprocessing.py > preprocessing_logs 2>&1
import json
import pandas as pd
import numpy as np
import yaml
import shutil
import subprocess
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.feature_selection import mutual_info_regression
from itertools import combinations
from statsmodels.formula.api import mixedlm
from Functions.plotting import plot_hist, plot_scatt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from Functions.preprocessing_functions import get_top_abs_correlations, remove_highly_correlated_columns, \
    check_condition, get_top_abs_phi, find_matching_rows, count_categories_to_file, log_drops
from Functions.stat_checks import is_binary, phi_coefficient
from fixed_params import (date_variables, keep_vars, remove_vars, dv_t1_name, school_track, target_id,
                          institution_id, dob_var, parent_duplicate_vars)
from Functions.gen_data import add_noise
from Functions.subsets import cfg_get
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns
#  -----------------------------------------------------------------------------------------
with open("configs/preprocessing.yaml", "r") as f:
    cfg = yaml.safe_load(f) or {}
with open("configs/base.yaml", "r") as f:
    cfg_base = yaml.safe_load(f) or {}
#  -----------------------------------------------------------------------------------------
var_info_sheet = cfg_get(cfg_base, ["paths", "var_info_csv"])
data_raw = cfg_get(cfg_base, ["paths", "data_raw"])

outputs_folder = cfg_get(cfg, ["paths", "outputs_folder"])
missing_thresh_col = cfg_get(cfg, ["preprocessing", "missing_thresh_col"])
missing_thresh_row = cfg_get(cfg, ["preprocessing", "missing_thresh_row"])
variance_feature_selection_threshold = cfg_get(cfg, ["preprocessing", "variance_feature_selection_threshold"])
smallest_category_count = cfg_get(cfg, ["preprocessing", "smallest_category_count"])
IV_cor_threshold = cfg_get(cfg, ["preprocessing", "IV_cor_threshold"])

run_R = cfg_get(cfg, ["switches", "run_R"])
use_synthetic_y = cfg_get(cfg, ["switches", "use_synthetic_y"])
create_corr_drop_list = cfg_get(cfg, ["switches", "create_corr_drop_list"])
check_mutual_information = cfg_get(cfg, ["switches", "check_mutual_information"])
merge_categories = cfg_get(cfg, ["switches", "merge_categories"])
create_new_vars = cfg_get(cfg, ["switches", "create_new_vars"])
remove_manual = cfg_get(cfg, ["switches", "remove_manual"])

# save a copy of the config params in outputs folder
Path(outputs_folder).mkdir(parents=True, exist_ok=True)
shutil.copy("configs/preprocessing.yaml", Path(outputs_folder) / "preprocessing_config.yaml")
#  -----------------------------------------------------------------------------------------
# check shape before R script:
df_pre_R = pd.read_csv(data_raw, low_memory=False, index_col=[0])
print(f"Data shape before R preprocessing: {df_pre_R.shape[1]} columns, {df_pre_R.shape[0]} rows")

## PREPROCESSING IN R (switched over to Python for simplicity)
if run_R == True:
    # r_script = 'Scripts/R_pre-processing.R'
    # # Run the R script
    # subprocess.call(['Rscript', r_script])
    from Scripts import R_preprocessing_script_into_Python as rprep
    rprep.main()
    print("Done running R pre-processing script (now in Python)!" \
    "------------------------------------------------------------")
#  -----------------------------------------------------------------------------------------
## IMPORT META DATA
# Import variable information & meta data:
var_info_all = pd.read_csv(var_info_sheet)
var_info = var_info_all[var_info_all["include as predictor"]==1] # data already filtered for this in R script

# SES/Edu:
edu_SES = pd.read_csv("Data/Meta/SES_Edu_classification.csv", encoding="ISO-8859-1", sep=',')
scale_drop = edu_SES[edu_SES['keep']==0]['scale']
scale_keep = edu_SES[edu_SES['keep']==1]['scale']

# Create a dictionary of variables_names: variable_descriptions
var_info_dict = dict(zip(var_info['var'], var_info['varname']))
var_info_all_dict = dict(zip(var_info_all['var'], var_info_all['varname']))
#  -----------------------------------------------------------------------------------------
## IMPORT PROCESSED DATA
# Import R pre-processed data:
df = pd.read_csv("Data/Preprocessed/df_R_to_python_processed.csv", low_memory=False)
print(f"Data shape after R preprocessing: {df.shape[1]} columns, {df.shape[0]} rows")

# Track all dropped variables and reasons
dropped_vars_reasons = {}

vars_to_drop = [v for v in parent_duplicate_vars if v in df.columns]
if vars_to_drop:
    print(f"Dropping parent-reported duplicate vars (keeping child-reported): {vars_to_drop}")
    log_drops(dropped_vars_reasons, vars_to_drop, "Parent duplicate (kept child version)")
    df.drop(vars_to_drop, axis=1, inplace=True)

# retain original school track info
school_track_dict = {}
for index, row in df.iterrows():
    key = row[target_id]
    value = row[school_track]
    school_track_dict[key] = value

# drop redundant edu/SES scales:
scale_drop = list(scale_drop)
matching_rows = find_matching_rows(column_data=list(var_info.varname), search_list=scale_drop)
scale_vars_to_drop = var_info.iloc[matching_rows]["var"]
print(f"num of Edu and SES scale vars dropped: {len(scale_vars_to_drop)}")
log_drops(dropped_vars_reasons, scale_vars_to_drop.tolist(), "Dropped Edu/SES scale")
df.drop(scale_vars_to_drop, axis=1, inplace=True)
print(f"num cols after dropping Edu and SES scales: {df.shape[1]}")
print(f"Dropping the following Edu and SES scale variables: {scale_vars_to_drop.tolist()}")

matching_rows_keep = find_matching_rows(column_data=list(var_info.varname), search_list=scale_keep)
scale_vars_to_keep = var_info.iloc[matching_rows_keep]["var"]
print(f"Keeping the following Edu and SES scale variables: {scale_vars_to_keep.tolist()}")
#  -----------------------------------------------------------------------------------------
## RECODE VARS
# Some variables are not truely ordinal, and need some additional pre-processing...
# in var_info.csv column "recode_does_not_apply", if a number, recode as -100 "not applicable"
recode_cols = var_info.loc[~var_info['recode_does_not_apply'].isna(), 'var']
recode_values = var_info[~var_info['recode_does_not_apply'].isna()]['recode_does_not_apply']
recode_cols_list = recode_cols.tolist()
recode_values_list = recode_values.tolist()
recode_dict = dict(zip(recode_cols_list, recode_values_list))
for col in df.columns:
    if col in recode_cols_list:
        value = recode_dict[col]
        df[col] = df[col].replace(value, -100)

# in var_info.csv column "recode_as_1", if there is a value here, it should be coded as 1
recode_1_cols = var_info.loc[~var_info['recode_as_1'].isna(), 'var']
recode_1_values = var_info[~var_info['recode_as_1'].isna()]['recode_as_1']
recode_1_cols_list = recode_1_cols.tolist()
recode_1_values_list = recode_1_values.tolist()
recode_1_dict = dict(zip(recode_1_cols_list, recode_1_values_list))
for col in df.columns:
    if col in recode_1_cols_list:
        value = recode_1_dict[col]
        df[col] = df[col].replace(value, 1)
#  -----------------------------------------------------------------------------------------
## CREATE NEW VARS
# Engineer new variables from existing, due to high multicollinearity between original variables:

if create_new_vars == True:
    # 1. Make two new variables for the teacher-state items based on e537030_D, e537110_g1, e537170_g1, e537200_D:
    # (a) the change of state for the teachers - whether they have ever changed the states (1) or not (0)
    newly_created_vars = set()
    teacher_loc_vars = ["e537030_D", "e537110_g1", "e537170_g1", "e537200_D"]
    df['teacher_changed_state'] = 0
    mask = (df[teacher_loc_vars] > 0).all(axis=1) &\
        (df[teacher_loc_vars].nunique(axis=1) > 1)
    df.loc[mask, 'teacher_changed_state'] = 1
    newly_created_vars.add("teacher_changed_state")

    # (b) "place of (predominant) teacher education"
    most_common_values = []
    for index, row in df[teacher_loc_vars].iterrows():
        row_numeric = pd.to_numeric(row, errors='coerce')
        row_numeric_positive = row_numeric[row_numeric > 0]
        if not row_numeric_positive.empty:
            most_common = row_numeric_positive.mode()
            most_common_values.append(most_common.iloc[0] if not most_common.empty else None)
        else:
            most_common_values.append(None)
    df['teacher_predominant_state'] = most_common_values

    # drop original columns:
    for var in teacher_loc_vars:
        log_drops(dropped_vars_reasons, [var], "Dropped after creating teacher state vars")
        df.drop(var, axis=1, inplace=True)
    newly_created_vars.add("teacher_predominant_state")

    # 2. Make a new variable for the stress factor of teachers, an average of ed1009b and ed1009c
    stress_factors = ["ed1009b", "ed1009c"]
    for i in stress_factors:
        df[i] = pd.to_numeric(df[i], errors='coerce')
    df["teacher_stress_factors_avg"] = df[df[stress_factors] > 0][stress_factors].apply(lambda x: np.nan if x.empty else x.mean(), axis=1)
    for var in stress_factors:
        log_drops(dropped_vars_reasons, [var], "Dropped after creating teacher_stress_factors_avg")
        df.drop(var, axis=1, inplace=True)
    newly_created_vars.add("teacher_stress_factors_avg")

    # 3. Make a new variable for the command of the other language, average of t41040b (speaking) and t41040a (comprehension)
    language_vars = ["t41040b", "t41040a"]
    for i in language_vars:
        df[i] = pd.to_numeric(df[i], errors='coerce')
    df["command_of_other_lang"] = df[df[language_vars] > 0][language_vars].apply(lambda x: np.nan if x.empty else x.mean(), axis=1)
    for var in language_vars:
        log_drops(dropped_vars_reasons, [var], "Dropped after creating command_of_other_lang")
        df.drop(var, axis=1, inplace=True)
    newly_created_vars.add("command_of_other_lang")

    # 4. Make a new variable for the language of media use, an average of t417030 (internet) and t417040 (text messages, email)
    language_media_vars = ["t417030", "t417040"]
    for i in language_media_vars:
        df[i] = pd.to_numeric(df[i], errors='coerce')
    df["lang_of_media"] = df[df[language_media_vars] > 0][language_media_vars].apply(lambda x: np.nan if x.empty else x.mean(), axis=1)
    for var in language_media_vars:
        log_drops(dropped_vars_reasons, [var], "Dropped after creating lang_of_media")
        df.drop(var, axis=1, inplace=True)
    newly_created_vars.add("lang_of_media")

    # 5. Make two new variables for the interaction language for parents and peers:
    # (a) an average of mother, father, and parents (t412010, t412020, t412060),
    fam_lang_vars = ["t412010", "t412020", "t412060"]
    for i in fam_lang_vars:
        df[i] = pd.to_numeric(df[i], errors='coerce')
    df["family_lang"] = df[df[fam_lang_vars] > 0][fam_lang_vars].apply(lambda x: np.nan if x.empty else x.mean(), axis=1)
    newly_created_vars.add("family_lang")

    # (b) an average of best friend and classmates (t412040, t412050)
    friend_lang_vars = ["t412040", "t412050"]
    for i in friend_lang_vars:
        df[i] = pd.to_numeric(df[i], errors='coerce')
    df["lang_of_friends"] = df[df[friend_lang_vars] > 0][friend_lang_vars].apply(lambda x: np.nan if x.empty else x.mean(), axis=1)
    # NOTE. The item of siblings (t412030) should be removed and it is on the modified list of the dropped vars.
    all_vars_rem = fam_lang_vars + friend_lang_vars + ["t412030"]
    for var in all_vars_rem:
        log_drops(dropped_vars_reasons, [var], "Dropped after creating family_lang/lang_of_friends")
        df.drop(var, axis=1, inplace=True)
    newly_created_vars.add("lang_of_friends")
    # Add newly created vars to keep list
    keep_vars.extend(list(newly_created_vars))

    # Save keep_vars names:
    keep_df = pd.DataFrame()
    keep_df['Keep Variables'] = keep_vars
    keep_df['Names'] = [var_info_all_dict[var] if var in var_info_all_dict else "created" for var in keep_vars]
    keep_df.to_csv("Data/Meta/keep_var_info.csv")
#  -----------------------------------------------------------------------------------------
## MERGE CATEGORIES
# e537161_g1: Subject combination (1st subject; study area) ...
# make one "religious/theology" cat.

if merge_categories == True:
    religious_codes = [2, 3]
    new_religion_value = 1001
    df['e537161_g1'] = df['e537161_g1'].replace(religious_codes, new_religion_value)
    # create one "fine art, art history, design, performing arts, film" cat.
    art_codes = [75, 76, 77, 74]
    new_art_value = 1002
    df['e537161_g1'] = df['e537161_g1'].replace(art_codes, new_art_value)
    # combine to create "comp science, math, engineering" cat.
    old_codes = [37, 38, 61]
    new_code = 1003
    df['e537161_g1'] = df['e537161_g1'].replace(old_codes, new_code)
    # make one "econ + industrial engineering with a focus on economics" cat.
    econ_codes = [30, 31]
    new_econ_value = 1004
    df['e537161_g1'] = df['e537161_g1'].replace(econ_codes, new_econ_value)
    # merge to one "social and education science" cat.
    old_codes = [16, 26]
    new_code = 1005
    df['e537161_g1'] = df['e537161_g1'].replace(old_codes, new_code)
    # make one "Slavic Studies, Baltic Studies, Finno-Ugrian Studies + Romance" cat.
    old_codes = [11, 12]
    new_code = 1006
    df['e537161_g1'] = df['e537161_g1'].replace(old_codes, new_code)

    # e537162_g1: Subject combination (2nd subject; study area)
    # make one "religious/theology" cat.
    religious_codes = [2, 3]
    new_religion_value = 1001
    df['e537162_g1'] = df['e537162_g1'].replace(religious_codes, new_religion_value)
    # create one "fine art, art history, design, performing arts, film" cat.
    art_codes = [75, 76, 77, 74]
    new_art_value = 1002
    df['e537162_g1'] = df['e537162_g1'].replace(art_codes, new_art_value)
    # combine to create "comp science, math, engineering" cat.
    old_codes = [37, 38, 61]
    new_code = 1003
    df['e537162_g1'] = df['e537162_g1'].replace(old_codes, new_code)
    # make one "econ + industrial engineering with a focus on economics" cat.
    econ_codes = [30, 31]
    new_econ_value = 1004
    df['e537162_g1'] = df['e537162_g1'].replace(econ_codes, new_econ_value)
    # merge to one "social and education science" cat.
    old_codes = [16, 26]
    new_code = 1005
    df['e537162_g1'] = df['e537162_g1'].replace(old_codes, new_code)
    # make one "Slavic Studies, Baltic Studies, Finno-Ugrian Studies + Romance" cat.
    old_codes = [11, 12]
    new_code = 1006
    df['e537162_g1'] = df['e537162_g1'].replace(old_codes, new_code)

    # p400500_g2: Missing/contradicting imformation about country of birth for generation status
    # create one cat, info for "one or both parents unknown"
    old_codes = [5, 6]
    new_code = 1007
    df['p400500_g2'] = df['p400500_g2'].replace(old_codes, new_code)

    # p727001_D: Recommendation secondary school or course of education, which ones? (simplified)
    # create one "other" cat.
    old_codes = [4, 5, 6, 11, 13]
    new_code = 1008
    df['p727001_D'] = df['p727001_D'].replace(old_codes, new_code)

    # p731813: (Highest) professional qualification Respondent
    #  create new "degree/bachelor" cat.
    old_codes = [16, 15, 12, 8]
    new_code = 1009
    df['p731813'] = df['p731813'].replace(old_codes, new_code)
    # create new "civil service/public admin" cat
    old_codes = [3, 13]
    new_code = 1010
    df['p731813'] = df['p731813'].replace(old_codes, new_code)
    #  create "other" cat
    old_codes = [7, 21, 17]
    new_code = 1011
    df['p731813'] = df['p731813'].replace(old_codes, new_code)

    # p731905: Professional status Respondent
    # add soldier to group (civil servant, including judge, excluding soldier) now "including soldier"
    old_codes = [3, 4]
    new_code = 1012
    df['p731905'] = df['p731905'].replace(old_codes, new_code)

    # p731110: Respondent's marital status
    # create new "married or civil partnership" cat.
    old_codes = [1, 6]
    new_code = 1013
    df['p731110'] = df['p731110'].replace(old_codes, new_code)

    # p731955: Professional status Partner
    # add soldier to group (civil servant, including judge, excluding soldier) now "including soldier"
    old_codes = [3, 4]
    new_code = 1014
    df['p731955'] = df['p731955'].replace(old_codes, new_code)
    # create new cat. "freelance or assisting family member"
    old_codes = [6, 7]
    new_code = 1015
    df['p731955'] = df['p731955'].replace(old_codes, new_code)

    # t400500_g2v1: Missing/contradicting imformation about country of birth for generation status
    # create one group "unknown"
    old_codes = [2, 9]
    new_code = 1016
    df['t400500_g2v1'] = df['t400500_g2v1'].replace(old_codes, new_code)

    # p400500_g2v1: Missing/contradicting imformation about country of birth for generation status
    # create one group "unknown for one or two grandparents"
    old_codes = [5, 6]
    new_code = 1017
    df['p400500_g2v1'] = df['p400500_g2v1'].replace(old_codes, new_code)

    # t400500_g2: Missing/contradicting imformation about country of birth for generation status
    # create one group "unknown"
    old_codes = [2, 9]
    new_code = 1018
    df['t400500_g2'] = df['t400500_g2'].replace(old_codes, new_code)

    # t269000: Student: sport activity: where/what ...
    # merge "Volkshochschule and school"
    old_codes = [4, 2]
    new_code = 1019
    df['t269000'] = df['t269000'].replace(old_codes, new_code)

    # t731130: Role of mother ...
    # merge all "non-biological mother"
    old_codes = [4, 2, 3, 5, 6]
    new_code = 1020
    df['t731130'] = df['t731130'].replace(old_codes, new_code)

    # t731140: Role of father ...
    # merge all "non-biological father"
    old_codes = [4, 2, 3, 5, 6]
    new_code = 1021
    df['t731140'] = df['t731140'].replace(old_codes, new_code)

#  -----------------------------------------------------------------------------------------
## DROP STUDENT TYPES
# a) drop students who don't aspire to finish school
rows_before = df.shape[0]
df = df[df['t31035a'] != 4]
rows_after = df.shape[0]
rows_dropped = rows_after - rows_before
print(f"Rows dropped where students dont aspire to finish school: {rows_dropped}")

# b) drop students in special needs school
rows_before = df.shape[0]
df = df[df['p727001_D'] != 12]
rows_after = df.shape[0]
rows_dropped = rows_after - rows_before
print(f"Rows dropped where students in a special needs school: {rows_dropped}")

#  -----------------------------------------------------------------------------------------
## REMOVE REDUNDANT VARS
# Remove ID variables (except target ID_t, that gets removed at end):
id_vars = ["cohort", "ID_cc", "ID_cm", "ID_tg_w1", "ID_e", "ID_cg"]
print("Dropping ID variables which are not needed for prediction and may cause data leakage...")
for var in id_vars:
    if var in df.columns:
        log_drops(dropped_vars_reasons, [var], "Dropped ID variable")
    df.drop(var, axis=1, inplace=True)
print(f"num cols after dropping {len(id_vars)} ID variables: {df.shape[1]}")

# Remove unwanted vars (duplicates and high multicollinearity):
print("Dropping variables which have been manually defined in the fixed_params.py file (redundant information)...")

if remove_manual == True:
    count = 0
    for var in remove_vars:
        if var in df.columns:
            print('dropping ' + var)
            log_drops(dropped_vars_reasons, [var], "Dropped manual remove_vars")
            df.drop(var, axis=1, inplace=True)
            count += 1
    print(f"num cols after dropping {count} manually defined redundant variables: {df.shape[1]}")

else:
    # just remove IDs and redundant dob vars:
    remove_vars_basic = ["Unnamed:0", # Incase index creeps in
                         "t262000_g1", # Type of sports (too many categories, not relevant)
                         "t70004m", # dob month (redundant info)
                        "t70004y", # dob year (redundant info)
                        "t70004", # dob date (redundant info)
                        "tx8050y", # redundant dob var year
                        "tx8050m", # redundant dob var month
                        "tx8050"] # redundant dob var date
    for var in remove_vars_basic:
        if var in df.columns:
            print('dropping ' + var)
            log_drops(dropped_vars_reasons, [var], "Dropped basic redundant var")
            df.drop(var, axis=1, inplace=True)
    print(f"num cols after dropping {len(remove_vars_basic)} basic redundant ID and dob variables: {df.shape[1]}")

#  -----------------------------------------------------------------------------------------
# COUNT MISSING
# for now, code all -100, -200, and -300 as missing
missing_values = [-100, -200, -300]

# Convert all values to strings
df_str = df.astype(str)

# Create a new DataFrame to store the counts
result_df = pd.DataFrame(index=df_str.columns, columns=missing_values)

# Loop through each column and count occurrences of each missing value
for col in df_str.columns:
    total_data_points = len(df[col])
    for missing_value in missing_values:
        count = (df_str[col] == str(missing_value)).sum()
        percentage = round((count / total_data_points) * 100, 2)
        result_df.at[col, missing_value] = percentage

# Convert the counts to numeric values
result_df = result_df.apply(pd.to_numeric, errors='coerce')

# Fill NaN values with 0
result_df.fillna(0, inplace=True)

# Add mean and save
result_df.loc['Average %'] = result_df.mean()
result_df.to_csv(f"{outputs_folder}/missing/percentages_missing_types.csv")

# Change all values of -100, -200, and -300 as NAs:
missing_values = [-100, '-100', -200, '-200', -300, '-300']

for val in missing_values:
    df.replace(to_replace=val, value=np.nan, inplace=True)

# Produce descriptives on missingness (counts and percentages) for each var (all types of missing)
missing_sum = pd.DataFrame(df.isna().sum())
missing_perc = round((missing_sum / len(df)) * 100, 2)
missing_summary = pd.concat([missing_sum, missing_perc], axis=1)
missing_summary.reset_index(inplace=True)
missing_summary.columns = ["Variable", "Missing_Sum", "Missing_Perc"]
missing_summary.sort_values(by="Missing_Sum", ascending=False, inplace=True)
missing_summary.to_csv(f"{outputs_folder}/missing/summary.csv")

# plot distribution of missingness (all types of missing):
plot_hist(save_name="cols_missing", x=missing_summary['Missing_Perc'],
          save_path=f"{outputs_folder}/missing/",
          title="Missing % of columns", bins=50,
          xlab="% missing")

# drop cols in df where > X% missing (all types):
drop_cols = missing_summary["Variable"][missing_summary["Missing_Perc"]>missing_thresh_col*100]
print(f"Dropping {len(drop_cols)} columns, where missingness >{missing_thresh_col*100}%")
count = 0
for col in drop_cols:
    if col != dv_t1_name:
        log_drops(dropped_vars_reasons, [col], f"Missing >{missing_thresh_col*100}% threshold")
        df.drop(col, axis=1, inplace=True)
        count += 1
drop_cols.to_csv(f"{outputs_folder}/missing/dropped_cols_{missing_thresh_col*100}%_miss.csv")
print(f"Dropped {count} columns where missingness >{missing_thresh_col*100}%")
print(f"New data shape after dropping columns >{missing_thresh_col*100}% missing: {df.shape[0]} rows, {df.shape[1]} columns")
#  -----------------------------------------------------------------------------------------
## ROW DROPS
# drop all rows where no DV at Time 1
rows_before = df.shape[0]
df.dropna(subset=[dv_t1_name], inplace=True, axis=0)
rows_after = df.shape[0]
rows_dropped = rows_before - rows_after
print(f"Rows dropped where no DV information at time 1: {rows_dropped}")
print(f"New data shape after dropping rows with no DV at time 1: {df.shape[0]} rows, {df.shape[1]} columns")

# Drop rows where > X% missing:
rows_before = df.shape[0]
threshold = len(df.columns) * missing_thresh_row  # X% of the columns
df.dropna(thresh=threshold, inplace=True)
rows_after = df.shape[0]
rows_dropped = rows_before - rows_after
print(f"Rows dropped where >{missing_thresh_row*100}% missing: {rows_dropped}")
print(f"New data shape after dropping rows >{missing_thresh_row*100}% missing: {df.shape[0]} rows, {df.shape[1]} columns")
# ------------------
# Final % of data missing descriptives:

# Calculate the overall percentage of missing values in the DataFrame
overall_missing_percentage = df.isnull().sum().sum() / df.size * 100
overall_missing_percentage = round(overall_missing_percentage, 2)
print("\nOverall percentage of missing values in the DataFrame:", overall_missing_percentage)

# Check the shape and overall missing values
print("Shape of DataFrame:", df.shape)
print("Total number of missing values:", df.isnull().sum().sum())
print("Calculated overall missing percentage:", overall_missing_percentage)

# Check the mean missingness for each column
missing_percentages = df.isnull().mean() * 100
missing_percentages = missing_percentages.round(2)
missing_percentages = missing_percentages.sort_values(ascending=False)
print("Missing percentages per column (top 10 highest):\n", missing_percentages.head(10))

# Plotting the distribution of missingness
missing_percentages = df.isnull().mean() * 100

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.xticks([])
plt.xlabel("Variables")
plt.ylabel("Respondents")
plt.title('Distribution of Missing Values')
plt.savefig(f"{outputs_folder}/missing/final_distribution_missingness_heatmap.png")
plt.clf()
plt.cla()
plt.close()

#  -----------------------------------------------------------------------------------------
## DATE VARIABLES
# Handle date variables...
df_orig = df.copy()

# First, check dates are merged:
merge_dates = list(var_info["var"][var_info["date_merge"] == "y"])
d = [var for var in merge_dates if var in df.columns]
if len(d) > 0:
    print(f"Need to merge variables: {d}")

# Change dates to days since dob
dob_variable = pd.to_datetime(df[dob_var], format='%Y-%m-%d', errors='coerce')

for col in date_variables:
    if col in list(df.columns):
        df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')
        df[col] = df[col] - dob_variable
        df[col] = df[col].dt.days

# Convert dob var to no of days (at the date of target survey: tx8600)
df[dob_var] = pd.to_datetime(df[dob_var], format='%Y-%m-%d', errors='coerce')
date_of_Target_survey = pd.to_datetime(df["tx8600"], format='%Y-%m-%d', errors='coerce')
df["dob_in_days_at_Target_survey"] = (date_of_Target_survey - df[dob_var]).dt.days
log_drops(dropped_vars_reasons, [dob_var], "Dropped raw dob after deriving dob_in_days_at_Target_survey")
df.drop(dob_var, axis=1, inplace=True)
log_drops(dropped_vars_reasons, ["tx8600"], "Dropped target survey date after deriving dob_in_days_at_Target_survey")
df.drop("tx8600", axis=1, inplace=True) # -> date of target questionnaire, used to calculate dob in days
print(f"Dropped original date variables and created new variables for days since dob for each date variable and dob in days at target survey.")
print(f"New data shape after processing date variables: {df.shape[0]} rows, {df.shape[1]} columns")
#  -----------------------------------------------------------------------------------------
## ASSIGN DATA TYPES
# Specify variable type:
categorical_vars = list(var_info["var"][var_info["final classification"] == "categorical"])
effect_coded_vars = list(var_info["var"][var_info["final classification"] == "effect code"])
continous_vars = list(var_info["var"][var_info["final classification"] == "continuous"])

categorical_vars = [var for var in categorical_vars if var in df.columns]
continous_vars = [var for var in continous_vars if var in df.columns]
effect_coded_vars = [var for var in effect_coded_vars if var in df.columns]

# Find columns in the DataFrame that are not in any of the three lists
df_columns_set = set(df.columns)
sorted_vars_set = set(categorical_vars + effect_coded_vars + continous_vars + date_variables)
remaining_columns = df_columns_set.difference(sorted_vars_set)
remaining_columns = [var for var in remaining_columns if var in df.columns]

# Convert columns
df[categorical_vars] = df[categorical_vars].astype('category')
df[effect_coded_vars] = df[effect_coded_vars].astype('category')
df[continous_vars] = df[continous_vars].astype('float')
df[remaining_columns] = df[remaining_columns].astype('float') # assume remaining columns are continuous

all_cat_vars = categorical_vars + effect_coded_vars

categorical_features = pd.Series(all_cat_vars)
categorical_features.to_csv("Data/Meta/categorical_variables.csv")
#  -----------------------------------------------------------------------------------------
## LOW VARIANCE
# Remove any variable with low variance (< threshold)
cols_before = df.shape[1]
selector = VarianceThreshold(variance_feature_selection_threshold)
selector.fit(df)

# Get mask of selected features
selected_mask = selector.get_support()
# Get names of dropped columns
dropped_columns = df.columns[~selected_mask]

# Filter the DataFrame with selected features
selected_features = selector.transform(df)
selected_columns = df.columns[selected_mask]
df_selected = pd.DataFrame(selected_features, columns=selected_columns)
cols_after = df_selected.shape[1]
num_cols_dropped = len(dropped_columns)
print(f"Columns dropped where low variance: {num_cols_dropped}")

# save the dropped col info
drop_col_list = list(df[dropped_columns].columns)
droped_col_df = pd.DataFrame(drop_col_list, columns=["Variable"])
save_path = f"{outputs_folder}/Variance_dropped_variables/"
droped_col_df.to_csv(save_path+"dropped_low_variance_{}.csv".format(variance_feature_selection_threshold))
log_drops(dropped_vars_reasons, drop_col_list, f"Variance <{variance_feature_selection_threshold} threshold")
df = df_selected.copy()
print(f"New data shape after dropping low variance variables: {df.shape[0]} rows, {df.shape[1]} columns")
#  -----------------------------------------------------------------------------------------
## MULTICOLLINEARITY

binary_vars = [col for col in df.columns if is_binary(df[col])]
non_binary_vars = [col for col in df.columns if col not in binary_vars]

if create_corr_drop_list == True:
    # Check multicollinearity:
    print(f"Checking for multicollinearity... No. columns: {df.shape[1]}")

    # 1. Get Phi pairs above threshold
    phi_coef = phi_coefficient(df[binary_vars])
    phi_coef.to_csv(f"{outputs_folder}/Phi_values/phi_matrix_before_changes.csv")

    # compare to correlations:
    corr_m_bin = round(df[binary_vars].corr(), 2)
    corr_m_bin.to_csv(f"{outputs_folder}/IV_correlations/BINARY_correlation_matrix_before_changes.csv")

    phi_above_threshold, all_phi = get_top_abs_phi(df, threshold=IV_cor_threshold, phi_coefs=phi_coef)
    phi_above_threshold.to_csv(f"{outputs_folder}/Phi_values/phi_pairs_above_{IV_cor_threshold}_before.csv")

    # 2. Get pairs of continous correlated variables above threshold
    corr_m = round(df[non_binary_vars].corr(), 2)
    corr_m.to_csv(f"{outputs_folder}/IV_correlations/CONTIN_correlation_matrix_before_changes.csv")

    cors_above_threshold_contin, all_cors = get_top_abs_correlations(df, cor_m=corr_m, threshold=IV_cor_threshold)
    cors_above_threshold_contin.to_csv(f"{outputs_folder}/IV_correlations/CONTIN_correlated_pairs_above_{IV_cor_threshold}_before.csv")

    # plot continuous correlated pairs where r >=0.9 and save
    cors_to_plot = cors_above_threshold_contin[(cors_above_threshold_contin["Corr"] >= 0.90) & (cors_above_threshold_contin["Corr"] < 1)]
    cor_plot_save_path = f"{outputs_folder}/IV_correlations/plots_pearson_r/"
    # todo: some vars are coded as continuous when binary (e.g., gender, state)

    for index, row in cors_to_plot.iterrows():
        xvar = row["Var1"]
        x = df[xvar]
        if xvar in var_info_dict.keys():
            xlab = f"{xvar}: {var_info_dict[xvar]}"
        else:
            xlab = xvar
        yvar = row["Var2"]
        y = df[yvar]
        if yvar in var_info_dict.keys():
            ylab = f"{yvar}: {var_info_dict[yvar]}"
        else:
            ylab = yvar
        plot_scatt(x=x, y=y, xlab=xlab, ylab=ylab, save_path=cor_plot_save_path,
                   save_name=f"{xvar}_{yvar}", jitter=True, fontsize=8)

    # plot distribution of corrs
    plot_hist(fig_size=(5, 5),
              bins=10,
              fontsize=12,
              x=all_cors["Corr"],
              title="Correlations between independent variables",
              xlab="Pearson r",
              ylab="Frequency",
              save_path=f"{outputs_folder}/IV_correlations/",
              save_name="plot_iv_corr_hist")

    # 3. Continuous and binary - correlations
    cors_above_threshold, all_cors = get_top_abs_correlations(df, threshold=IV_cor_threshold)
    # add var_info and save:
    cors_above_threshold['Var1 name'] = [var_info_all_dict[var] if var in var_info_all_dict else "created" for var in
                                          cors_above_threshold['Var1']]
    cors_above_threshold['Var2 name'] = [var_info_all_dict[var] if var in var_info_all_dict else "created" for var in
                                          cors_above_threshold['Var2']]
    cors_above_threshold = cors_above_threshold[["Var1", "Var1 name", "Var2", "Var2 name", "Corr"]]
    if create_new_vars == True:
        cors_above_threshold.to_csv(f"{outputs_folder}/IV_correlations/ALL_correlated_pairs_above_{IV_cor_threshold}_before_wINFO.csv")
    if create_new_vars == False:
        cors_above_threshold.to_csv(f"{outputs_folder}/IV_correlations/ALL_correlated_pairs_above_{IV_cor_threshold}_before_wINFO_original_vars.csv")


    # remove highly correlated variables in a way which minimises number of cols to drop (continous and binary)
    df_filtered, columns_to_remove = remove_highly_correlated_columns(df, threshold=IV_cor_threshold, keep_vars=keep_vars)
    columns_to_remove_list = list(columns_to_remove)
    # subset var_info to inspected dropped vars:
    dropped_var_info = var_info[var_info['var'].isin(columns_to_remove)].reset_index()
    dropped_var_info.to_csv(f"{outputs_folder}/IV_correlations/dropped_vars_above_{IV_cor_threshold}_wINFO.csv")
    # save drop list
    drop_list_series = pd.DataFrame(columns_to_remove_list, columns=["Variable"])
    drop_list_series.to_csv(f"{outputs_folder}/IV_correlations/dropped_vars_above_{IV_cor_threshold}.csv")
    log_drops(dropped_vars_reasons, columns_to_remove_list, f"Multicollinearity >{IV_cor_threshold} threshold")

if create_corr_drop_list == False:
    columns_to_remove = pd.read_csv(f"{outputs_folder}/IV_correlations/dropped_vars_above_{IV_cor_threshold}.csv", index_col=[0])
    columns_to_remove.columns = ['Variable']
    columns_to_remove_list = list(columns_to_remove['Variable'])
    # Remove the identified columns from the DataFrame
    df_filtered = df.drop(columns=columns_to_remove_list)
    log_drops(dropped_vars_reasons, columns_to_remove_list, f"Multicollinearity >{IV_cor_threshold} threshold")

# define df as filtered df
df = df_filtered.copy()

print(f"Columns dropped due to IV correlations >{IV_cor_threshold}: {len(columns_to_remove)}")
print(f"New data shape after dropping correlated variables: {df.shape[0]} rows, {df.shape[1]} columns")

# recheck pairs:
cors_above_threshold2, all_cors2 = get_top_abs_correlations(df, threshold=IV_cor_threshold)
if create_new_vars == True:
    cors_above_threshold2.to_csv(f"{outputs_folder}/IV_correlations/ALL_correlated_pairs_above_{IV_cor_threshold}_after.csv")
if create_new_vars == False:
    cors_above_threshold2.to_csv(f"{outputs_folder}/IV_correlations/ALL_correlated_pairs_above_{IV_cor_threshold}_after_original_vars.csv")
#  -----------------------------------------------------------------------------------------
## MUTUAL INFORMATION
if check_mutual_information == True:
    print("calculating mutual information...")
    df_fill = df.fillna(-999) # fill NAs with -999 as will in tree-based models with no imputation

    # Calculate mutual information between all pairs of variables
    mi_scores = {}
    for col1, col2 in combinations(df_fill.iloc[:,2:].columns, 2):
        # print(f'Calculating mutual information between {col1} and {col2}')
        mi_score = mutual_info_regression(df_fill[[col1]], df_fill[col2])[0]
        mi_scores[(col1, col2)] = mi_score
        mi_scores[(col2, col1)] = mi_score  # Mutual information is symmetric

    # Sort the mutual information scores in descending order
    sorted_mi_scores = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)

    # Create a DataFrame from the sorted scores
    unpacked_data = [(x[0], x[1], z) for (x, z) in sorted_mi_scores]
    mi_df = pd.DataFrame(unpacked_data, columns=['Variable 1', 'Variable 2', 'Mutual Information'])

    # add var_info and save:
    mi_df['Var1 name'] = [var_info_all_dict[var] if var in var_info_all_dict else "created" for var in
                                          mi_df['Variable 1']]
    mi_df['Var2 name'] = [var_info_all_dict[var] if var in var_info_all_dict else "created" for var in
                                          mi_df['Variable 2']]
    mi_df = mi_df[["Variable 1", "Var1 name", "Variable 2", "Var2 name", "Mutual Information"]]

    # Save the DataFrame to a CSV file
    mi_df.to_csv(f'{outputs_folder}/mutual_information/mutual_information_scores.csv', index=False)

elif check_mutual_information == False:
    mi_df = pd.read_csv(f'{outputs_folder}/mutual_information/mutual_information_scores.csv')

# Plot MI scores above 1.3:
mi_plot_save_path = f'{outputs_folder}/mutual_information/Plots/'
mi_df = mi_df[mi_df['Mutual Information'] > 1.3]
for index, row in mi_df.iterrows():
    xvar = row["Variable 1"]
    x = df[xvar]
    if xvar in var_info_dict.keys():
        xlab = f"{xvar}: {var_info_dict[xvar]}"
    else:
        xlab = xvar
    yvar = row["Variable 2"]
    y = df[yvar]
    if yvar in var_info_dict.keys():
        ylab = f"{yvar}: {var_info_dict[yvar]}"
    else:
        ylab = yvar
    plot_scatt(x=x, y=y, xlab=xlab, ylab=ylab, save_path=mi_plot_save_path,
               save_name=f"{xvar}_{yvar}", jitter=True, fontsize=8)
#  -----------------------------------------------------------------------------------------
## GET DROP INFO - Create a dataframe of all vars that get dropped and reason why
drop_records = []
for var, reasons in dropped_vars_reasons.items():
    drop_records.append({
        "Variable": var,
        "Drop_Reason": "; ".join(sorted(reasons))
    })

all_dropped_vars_df = pd.DataFrame(drop_records)
print(f"Total variables dropped: {all_dropped_vars_df.shape[0]}")
# Split and create new column with original var (prefix)

# Creating a new column based on the conditions
all_dropped_vars_df['Original_Var'] = all_dropped_vars_df['Variable'].apply(check_condition)

# Add variable info from original col
all_dropped_vars_df_info = pd.merge(all_dropped_vars_df, var_info[['var', 'varname', 'Variable_Group']],
                     left_on='Variable', right_on='var', how='left')

if create_new_vars == True:
    all_dropped_vars_df_info.to_csv(f"{outputs_folder}/dropped_variables_miss{missing_thresh_col*100}%_ivcor{IV_cor_threshold}_var_{variance_feature_selection_threshold}.csv")
if create_new_vars == False:
    all_dropped_vars_df_info.to_csv(f"{outputs_folder}/dropped_variables_miss{missing_thresh_col*100}%_ivcor{IV_cor_threshold}_var_{variance_feature_selection_threshold}_original_vars.csv")
#  -----------------------------------------------------------------------------------------
## ICC - compute the inter-class correlation (institution-level) of outcome for residual scores after regressing out school track

icc_X = df[[school_track] + [institution_id]]
icc_y = df[dv_t1_name]

# Create and fit the linear regression model
X_m = icc_X.loc[:, icc_X.columns]
model = LinearRegression()
model.fit(X_m, icc_y)

# Predict the values
icc_y_pred = model.predict(X_m)

# Calculate the residuals
residuals = icc_y - icc_y_pred

# Construct data
groups = icc_X[institution_id]
data = pd.concat([residuals, groups], axis=1)

# Calculate inter-class correlation using mixed lm (one student in only one school, doesn't matter if unequal number of students in each school)
formula = dv_t1_name + ' ~ 1'

# Fit the mixed effects model
model = mixedlm(formula, data=data, groups=institution_id, re_formula='1', missing='drop')
result = model.fit()

# Extract the ICC value
icc = result.cov_re[institution_id] / (result.cov_re[institution_id] + result.scale)
print('ICC: {}'.format(round(icc, 2)))

# Remove IDs
log_drops(dropped_vars_reasons, [target_id, institution_id], "Dropped IDs before final modeling")
df.drop([target_id, institution_id], axis=1, inplace=True)

#----------------------------------------------------------------------------------------
# Check category distributions across df
save_path = outputs_folder + "/category_distributions/"
categorical_features_in_data = [elem for elem in categorical_features if elem in list(df.columns)]
with open('Data/Meta/label_info_nested_dict.json', 'r') as file:
    cat_name_dict = json.load(file)
count_categories_to_file(df, output_file=save_path+"X_counts.txt", categorical_columns=categorical_features_in_data,
                             var_name_dict=var_info_dict, min_cat=smallest_category_count, cat_name_dict=cat_name_dict,
                             data_name="X")

# save final categorical features
categorical_features = [elem for elem in categorical_features if elem in list(df.columns)]
categorical_features_series = pd.DataFrame(categorical_features, columns=["Categorical_Variables"])
categorical_features_series.to_csv("Data/Meta/final_categorical_variables_after_preprocessing.csv")

#---------------------------- (only needed for the synthetic data) ----------------------------
if use_synthetic_y == True:
    print("Creating mock y variable for piloting...")
    # create y variable:
    print("Calculating new y variable...")
    dv_t1 = df[dv_t1_name]
    plot_hist(save_name="dv_t1", x=dv_t1, save_path=f"{outputs_folder}/histograms/",
            title="dv_t1", bins=50)

    # Predict y from X with fixed coefficients (b=1)
    y_pred = np.dot(dv_t1, 1)

    # Add noise to y_pred so that X predicts y with a given r2 (0.5)
    y, iters_count = add_noise(y_pred, 0.5)

    # Create and fit the linear regression model to check R2
    model = LinearRegression()
    dv_t1_2d = np.array(dv_t1).reshape(-1, 1)
    model.fit(np.array(dv_t1_2d), y)

    # Predict values
    y_pred = model.predict(dv_t1_2d)

    # Calculate R² score
    r2 = r2_score(y, y_pred)
    print(f"R-squared score: {round(r2, 2)}")

    plot_hist(save_name="y", x=y, save_path=f"{outputs_folder}/histograms/",
            title="y", bins=50)

    # check correlation between dv_t1 and y
    corr, _ = pearsonr(dv_t1, y)
    print("Pearson's r between DV at time 1 and y:", round(corr, 2))
    plot_scatt(x=dv_t1, y=y, save_path=f"{outputs_folder}/", save_name="scatter_dvt1_and_y",
            xlab="DV Time 1", ylab="DV Time 2 (generated)")

    # add y to data and save:
    y = pd.Series(y)
    y.name = "y"
    df.reset_index(inplace=True, drop=True)
    X_and_y = pd.concat([df, y], axis=1, join="inner")
    X_and_y.to_csv("Data/Preprocessed/X_and_y.csv")
    print(f"final data shape: {X_and_y.shape}")

print('preprocessing done')