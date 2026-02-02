print("Running R preprocessing file...")

# one time set-up
# Create a personal R library
dir.create("~/Library/R/4.5/library", recursive = TRUE, showWarnings = FALSE)

# Tell R to use it
.libPaths(c("~/Library/R/4.5/library", .libPaths()))

# options(repos = "https://cran.r-project.org/")
# install.packages("ggplot2", dependencies = TRUE)
# install.packages("stringr")
# install.packages("tidyr")
# install.packages("psych")

# packages ####
library(ggplot2)
library(stringr)
library(tidyr)
library(psych)

# functions ####
as_numeric_trycatch <- function(column) {
  out <- tryCatch(
    {as.numeric(column)},
    error=function(cond) {
      # Choose a return value in case of error
      return(toString(cond))
    },
    warning=function(cond) {
      # Choose a return value in case of warning
      return(toString(cond))
    }
  )    
  return(out)
}

# read in data ####
print(getwd())
df<-read.csv("Data/neps.csv")
variable_info<-read.csv("Data/Meta/variable_info_agreed_Jan_21_2026.csv", stringsAsFactors = F)

# subset to agreed variables ####
variables<-as.character(variable_info[variable_info$include.as.predictor==1 | variable_info$include.as.ID==1,]$var)
variables<-variables[!is.na(variables)]
df<-df[,variables]

# label missingness as either missing by design (-100) or random (-200) or editorial (-300) ####
## check that all variables are integer (often missingness can be entered oddly as text) ####
for(i in variables){
  if(typeof(as_numeric_trycatch(df[,i]))=="character"){
    print(paste0("check ", i))
  }
}

# Count specific values
values_to_count <- c('-54', '93', '-99')

# Convert the dataset to strings
df_str <- as.data.frame(lapply(df, as.character))

# Initialize a matrix to store percentages
percentages_matrix <- matrix(0, nrow = ncol(df_str), ncol = length(values_to_count),
                             dimnames = list(colnames(df_str), values_to_count))

# Loop through each column and value to calculate percentages
for (col in seq_along(df_str)) {
  total_data_points <- sum(!is.na(df_str[[col]]))

  for (i in seq_along(values_to_count)) {
    count <- sum(df_str[[col]] == values_to_count[i], na.rm = TRUE)
    percentages_matrix[col, i] <- round(count / total_data_points * 100, 2)
  }
}

# Convert the matrix to a data frame
percentages_df <- as.data.frame(percentages_matrix)

# Add a row for total percentages and save
average_percentages <- rowMeans(percentages_matrix, na.rm = TRUE)
percentages_df <- rbind(percentages_df, Average = average_percentages)

write.csv(percentages_df, "Outputs/missing/R_mising_-100_perc.csv")

## recode variables ####
# see NEPS manual (https://www.neps-data.de/Portals/0/NEPS/Datenzentrum/Forschungsdaten/SC6/13-0-0/SC6_13-0-0_DataManual.pdf)
# NOT APPLICABLE (-100)
# # NA  = not collected by design
# # -93 = does not apply

# RANDOM (-200) ... okay to impute
# # -94 = not reached
# # -95 = implausible value
# # -97 = refused
# # -98 = don't know
# # -20...-29 = various item-specific
# # -90 = unspecified missing
# # -91 = survey aborted
# # -92 = question erroneously not asked
# # -56 = not participated
# also including here -54 MISSING BY DESIGN as assume ranodm sample
# and -99 = filtered (same as missing by design)
# 
# EDITORIAL (-300)
# # -52 = implausible value removed
# # -53 = anonymised
# # -55 = not determinable

for(i in colnames(df)){
  df[df[,i] %in% c(-93) | is.na(df[,i]),i]<--100
  df[df[,i] %in% c(-99, -54, -94, -95, -97, -98, -90, -91, -92, -56, -29:-20),i]<--200
  df[df[,i] %in% c(-52, -53, -55),i]<--300
}

# take m and y date variables and combine to make a date/year in mmdd ####
monthvar<-variable_info[variable_info$date_merge=="y" & grepl("m\\b", variable_info$var),]$var

## recode the less specific month categories ####
# 21 Beginning of the year/Winter == Jan (01)
# 24 Spring/Easter == Apr (04)
# 27 Mid-year/Summer== July (07)
# 30 Fall == October (10)
# 32 End of the year == December (12)
correct_m<-data.frame(x=c(21, 24, 27, 30, 32), y=c(1, 4, 7, 10, 12))
for(i in monthvar){
  for(x in correct_m$x){
    if(dim(df[df[,i]==x & !is.na(df[,i]),])[1]!=0){
      df[df[,i]==x & !is.na(df[,i]),i]<-correct_m[correct_m$x==x,]$y
    }
  }
}

for(i in monthvar){
  newname<-gsub("m\\b", "", i) # make new variable (without m or y)
  yearvar<-paste0(newname, "y") # corresponding year variable
  
  eval(parse(text=paste0("df$", newname, "<-df$", i))) # copy variable across to preserve missingness
  
  # only replace for those falling as actual date (i.e., not missing) values
  df[df[,newname] %in% c(1:12),newname]<-as.character(as.Date(paste0("01/", sprintf("%02d", df[df[,newname] %in% c(1:12),i]), "/" ,df[df[,newname] %in% c(1:12),yearvar]), "%d/%m/%Y") )
  
  # remove separate m and y cols
  df[,yearvar]<-NULL
  df[,i]<-NULL
}

# aggregate variables (top-down decision) #####
# aggregates excluding missing (does not exclude if one value missing, only if all are)
# if all are missing for same reason, gives missing variable designation
# introduces new missing variable designation (-400)
# this is for an aggregated variable where missingness is due to different reasons (affecting different items)

# record alpha
aggregated_vars<-data.frame(var=unique(variable_info[variable_info$aggregate!="",]$aggregate), alpha=NA)

x<-1 # to report progress
for(i in aggregated_vars$var){
  for_agg<-(df[,colnames(df) %in% (variable_info[variable_info$aggregate==i,]$var)])
  
  for_agg_no_miss<-for_agg
  colnames(for_agg_no_miss)<-paste0(colnames(for_agg_no_miss), "_no_miss")
  # first remove missing value codes to NA
  for(j in 1:dim(for_agg_no_miss)[2]){
    for_agg_no_miss[for_agg_no_miss[,j] %in% c(-100,-200,-300),j]<-NA
  }
  
  # reverse code if required
  for(j in (variable_info[variable_info$aggregate==i,]$var)){
    if(variable_info[variable_info$var==j,]$reverse==1){
      for_agg_no_miss[,paste0(j, "_no_miss")]<-(variable_info[variable_info$var==j & variable_info$reverse==1,]$reverse_scale_max+1)-for_agg_no_miss[,paste0(j, "_no_miss")]
    }
  }
  
  # aggregate
  agg<-rowMeans(for_agg_no_miss)
  full_agg<-cbind(for_agg, for_agg_no_miss, agg)
  
  # reintegrate missingness information
  for(j in row.names(full_agg[is.na(full_agg$agg),])){
    if(length(unique(c(full_agg[j,colnames(for_agg)])[[1]]))==1){
      full_agg[j,]$agg<-full_agg[j,colnames(for_agg)[1]]
    }else{full_agg[j,]$agg<--400}
  }
  
  # make aggregated variable in df
  eval(parse(text=paste0("df$", i, "<-full_agg$agg")))
  
  # calculate alpha
  aggregated_vars[aggregated_vars$var==i,]$alpha<-psych::alpha(for_agg_no_miss)$total$raw_alpha
  
  # remove variables from df that are aggregated across
  df<-df[, !colnames(df) %in% colnames(for_agg)]
  print(paste0(i, ": ", x, "/", length(unique(variable_info[variable_info$aggregate!="",]$aggregate))))
  x<-x+1
}

## evaluate goodness of aggregation ####
aggregated_vars

write.csv(aggregated_vars, "Outputs/R_preprocessing/alpha_for_aggregated_vars.csv")

write.csv(df, "Data/Preprocessed/df_R_processed.csv")

dim(df)
print("finished running R preprocessing script")


