# fixed data coding parameters
import pandas as pd

# ---- meta data coding ----
var_info_sheet = "Data/Meta/variable_info_agreed_Feb_2_2026.csv"
target_id = "ID_t"
school_track = "tx80106"
institution_id = "ID_i"
dv_t1_name = "reg5_sc1" # name of dv at time 1
dob_var = "p70012"
date_variables = ["tx8602", "tx8611", "tx8620", "p40603", "p71202", "p71203", "tx8610",
                  "p72802", "p40003", "p40113", "p73191", "p73111", "p40403", "p73195"]

# ---- variable removal coding ----
remove_vars = [
# Used in creation of teacher_changed_state and teacher_predominant_state:
"e537030_D", # Federal state higher education entrance qualification (West/East)
"e537110_g1", # Place of study Teaching degree course (West/East)
"e537170_g1", # Location: passed the state examination (West/East)
"e537200_D", # Federal state second state examination (West/East)
# Multicollinearity drop (manual decisions)
"p731116", # Gender of parent respondents partner (multicollinearity)
"e537122", # Gender of parent respondents partner (multicollinearity)
"e53702y_D", # Teacher: Activity before teaching degree course - military/civil service (multicollinearity)
"e53710y_D", # Year Study start Teaching degree course (multicollinearity)
"e537150_D", # Year of state examination (multicollinearity)
"e229821_D", # Work at this school (years) (multicollinearity)
"p413000_g1D", # respondents first language (parent), highly correlated with child's first language
"p400000", #Respondent born in Germany?
"p400000_g1", # Country of birth Respondent (Germany/abroad; edited)
"p401110", # Citizenship Respondent German since birth
"p413040", #Ability to speak German Interviewed Parent (auto variable)
"t41010a_g2D", # Mother: Mother tongue (reference 1, simplified) -- use generated variable (t41010a_g1)
"t400240_g1D", # Mother's father: Country of birth (Germany/abroad)
"t400220_g1D", # Mother's mother: country of birth (Germany/abroad)
"t400070_g1D", # Mother: Country of birth (Germany/abroad)
"p404000", # German citizenship Partner
"p403000", # Country of birth Partner (Germany/abroad)
"p403000_g1", # Country of birth Partner (Germany/abroad; edited)
"p404010", # German citizenship Partner since birth
"t41012a_g2D", # Father: Mother tongue (Specification 1, simplified)
"t400280_g1D", # Father's father: Country of birth (Germany/abroad)
"t400260_g1D", # Father's mother: country of birth (Germany/abroad)
"t400090_g1D", #Father: Country of birth (Germany/abroad)
"teacher_predominant_state", #  Highly correlated with students' state of residence
# Redundant DOB info:
"t70004m", # dob month (redundant info)
"t70004y", # dob year (redundant info)
"t70004", # dob date (redundant info)
"tx8050y", # redundant dob var year
"tx8050m", # redundant dob var month
"tx8050",  # redundant dob var date
# Other:
"Unnamed:0", # Incase index creeps in
"t262000_g1"] # Type of sports (too many categories, not relevant)

# *Note that this list is not comprehensive; further variables are removed at other stages
# such as when creating new variables due to multicollinearity and iteratively removing high collinear variables

keep_vars = ["p731702", "e229820_D", "p751001_g1", "p410000_g1D", "p414040"] + [school_track] + [target_id] + [institution_id] #IDs removed later
            # ^ i.e., vars that should not be dropped due to multicollinearity or other reasons
