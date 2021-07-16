# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 11:33:12 2021

@author: BRSch
"""

import pandas as pd
import numpy as np

# Data is read from a local file
data = pd.read_csv("owid-covid-data.csv")


data = data[
    (data["iso_code"] == "FRA") | (data["iso_code"] == "USA")]
# data contains data from both the France and the USA.
data = data[["iso_code", "location", "date", "total_cases", "new_cases", "total_deaths", "new_deaths",
             "new_deaths_smoothed", "total_cases_per_million", "new_cases_per_million",
             "new_cases_smoothed_per_million", "total_deaths_per_million", "new_deaths_per_million",
             "new_deaths_smoothed_per_million", "reproduction_rate", "icu_patients", "icu_patients_per_million",
             "hosp_patients", "hosp_patients_per_million", "weekly_icu_admissions", "weekly_icu_admissions_per_million",
             "weekly_hosp_admissions", "weekly_hosp_admissions_per_million", "new_tests", "total_tests",
             "total_tests_per_thousand", "new_tests_per_thousand", "new_tests_smoothed",
             "new_tests_smoothed_per_thousand", "positive_rate",  "total_vaccinations",
             "people_vaccinated", "people_fully_vaccinated", "new_vaccinations", "new_vaccinations_smoothed",
             "total_vaccinations_per_hundred", "people_vaccinated_per_hundred", "people_fully_vaccinated_per_hundred",
             "new_vaccinations_smoothed_per_million", "stringency_index", "population"]]

data['date']= pd.to_datetime(data['date']).dt.date

# =============================================================================
# Vaccination data preparation
# =============================================================================

USA_data= data[data['iso_code']== 'USA']

USA_vaccination_data= USA_data[['date', 'people_fully_vaccinated_per_hundred']]

# Start the dataset at the first non-NaN value, assuming this is the point where they started vaccinating
first_vaccination= 358
USA_vaccination_data= USA_vaccination_data.iloc[first_vaccination:]

USA_vaccination_data= USA_vaccination_data.reset_index(drop= True)
# All other NaN values are substituted by the mean of the surrounding values,
# assuming this is representative since vaccination cover variables increase relatively constant (over small time scales)

USA_vaccination_data= USA_vaccination_data.fillna(0)
# NaN's set to 0 for easier computing

for i in range(len(USA_vaccination_data)):
    if USA_vaccination_data.iat[i, 1]== 0:
        j= 1
        while USA_vaccination_data.iat[i+j, 1]== 0:
            j+= 1
        prev_value= USA_vaccination_data.iat[i-1, 1]
        next_value= USA_vaccination_data.iat[i+j, 1]
        substitute= np.mean([prev_value, next_value])
        USA_vaccination_data.iat[i, 1]= substitute


FRA_data= data[data['iso_code']== 'FRA']

FRA_vaccination_data= FRA_data[['date', 'people_fully_vaccinated_per_hundred']]
FRA_vaccination_data= FRA_vaccination_data.reset_index(drop= True)
# Start the dataset at the first non-NaN value, assuming this is the point where they started vaccinating
first_vaccination= 358
last_vaccination= 524
FRA_vaccination_data= FRA_vaccination_data.iloc[first_vaccination:last_vaccination]

FRA_vaccination_data= FRA_vaccination_data.reset_index(drop= True)
# All other NaN values are substituted by the mean of the surrounding values,
# assuming this is representative since vaccination cover variables increase relatively constant (over small time scales)

FRA_vaccination_data= FRA_vaccination_data.fillna(0)

# NaN's set to 0 for easier computing


# substitute NA's within the data series with the mean of surrounding values
for i in range(len(FRA_vaccination_data)):
    if FRA_vaccination_data.iat[i, 1]== 0:
        prev_value= FRA_vaccination_data.iat[i-1, 1]
        next_value= FRA_vaccination_data.iat[i+1, 1]
        substitute= np.mean([prev_value, next_value])
        FRA_vaccination_data.iat[i, 1]= substitute
# =============================================================================
# Cases data preparation
# =============================================================================

USA_cases_data= USA_data[['date', 'new_cases']]
USA_cases_data= USA_cases_data.reset_index(drop= True)


# for column in USA_cases_data.columns:
#     print(USA_cases_data[USA_cases_data[column].isna()])
#
# This can be used to check where NA's are, but is commented out to prevent 
# clogging up the console print

# Since only the 1st row (i.e. start of measurements) contains a NaN value, we're going to assume it is 0
USA_cases_data= USA_cases_data.fillna(0)

# replacing negative values with absolute values as negative values are non-sensical,
# so we assume they should simply be not negative
for i in range(len(USA_cases_data)):
    if USA_cases_data.iat[i, 1]<0:
        USA_cases_data.iat[i, 1]= np.abs(USA_cases_data.iat[i, 1])

# Creating normalised case variables (per million for easy interpretation)
population_USA= USA_data.iat[0,-1]
new_cases_per_million= [value/population_USA*1000000 for value in USA_cases_data['new_cases']]
USA_cases_data['new_cases_per_million']= new_cases_per_million

USA_cases_data= USA_cases_data.merge(USA_vaccination_data, on= 'date')


FRA_cases_data= FRA_data[['date', 'new_cases']]
FRA_cases_data= FRA_cases_data.reset_index(drop= True)

# replacing negative values with absolute values as negative values are non-sensical,
# so we assume they should simply be not negative
for i in range(len(FRA_cases_data)):
    if FRA_cases_data.iat[i, 1]<0:
        FRA_cases_data.iat[i, 1]= np.abs(FRA_cases_data.iat[i, 1])

# for column in FRA_cases_data.columns:
#     print(FRA_cases_data[FRA_cases_data[column].isna()])
#
# This can be used to check where NA's are, but is commented out to prevent 
# clogging up the console print


# substitute NA's within the data series with the mean of surrounding values    
for i in range(len(FRA_cases_data)):
    if np.isnan(FRA_cases_data.iat[i, 1]):
        prev_value= FRA_cases_data.iat[i-1, 1]
        next_value= FRA_cases_data.iat[i+1, 1]
        substitute= np.mean([prev_value, next_value])
        FRA_cases_data.iat[i, 1]= substitute
    

# Creating normalised case variables (per million for easy interpretation)
population_FRA= FRA_data.iat[0,-1]
new_cases_per_million= [value/population_FRA*1000000 for value in FRA_cases_data['new_cases']]
FRA_cases_data['new_cases_per_million']= new_cases_per_million

FRA_cases_data= FRA_cases_data.merge(FRA_vaccination_data, on= 'date')
# =============================================================================
# Deaths data preparation
# =============================================================================

USA_deaths_data= USA_data[['date', 'total_deaths', 'new_deaths']]
USA_deaths_data= USA_deaths_data.reset_index(drop= True)


# for column in USA_deaths_data.columns:
#     print(USA_deaths_data[USA_deaths_data[column].isna()])
#
# This can be used to check where NA's are, but is commented out to prevent 
# clogging up the console print


# replacing negative values with absolute values as negative values are non-sensical,
# so we assume they should simply be not negative
for i in range(len(USA_deaths_data)):
    if USA_deaths_data.iat[i, 1]<0:
        USA_deaths_data.iat[i, 1]= np.abs(USA_deaths_data.iat[i, 1])
# Start the dataset at the first non-NaN value, since we cannot know the deaths before this point

first_death= 38
USA_deaths_data= USA_deaths_data.iloc[first_death:]

USA_deaths_data= USA_deaths_data.reset_index(drop= True)


# Creating normalised case variables
# Total deaths per hundred for easy interpretation
total_deaths_per_hundred= [value/population_USA*100 for value in USA_deaths_data['total_deaths']]
USA_deaths_data['total_deaths_per_hundred']= total_deaths_per_hundred
# New deaths per million for easy interpretation
new_deaths_per_million= [value/population_USA*1000000 for value in USA_deaths_data['new_deaths']]
USA_deaths_data['new_deaths_per_million']= new_deaths_per_million

USA_deaths_data= USA_deaths_data.merge(USA_vaccination_data, on= 'date')


FRA_deaths_data= FRA_data[['date', 'total_deaths', 'new_deaths']]
FRA_deaths_data= FRA_deaths_data.reset_index(drop= True)


# for column in FRA_deaths_data.columns:
#     print(FRA_deaths_data[FRA_deaths_data[column].isna()])
#
# This can be used to check where NA's are, but is commented out to prevent 
# clogging up the console print


# replacing negative values with absolute values as negative values are non-sensical,
# so we assume they should simply be not negative
for i in range(len(FRA_deaths_data)):
    if FRA_deaths_data.iat[i, 1]<0:
        FRA_deaths_data.iat[i, 1]= np.abs(FRA_deaths_data.iat[i, 1])

# Start the dataset at the first non-NaN value, since we cannot know the deaths before this point

first_death= 22
FRA_deaths_data= FRA_deaths_data.iloc[first_death:]

# Creating normalised case variables
# Total deaths per hundred for easy interpretation
total_deaths_per_hundred= [value/population_FRA*100 for value in FRA_deaths_data['total_deaths']]
FRA_deaths_data['total_deaths_per_hundred']= total_deaths_per_hundred
# New deaths per million for easy interpretation
new_deaths_per_million= [value/population_FRA*1000000 for value in FRA_deaths_data['new_deaths']]
FRA_deaths_data['new_deaths_per_million']= new_deaths_per_million

FRA_deaths_data= FRA_deaths_data.merge(FRA_vaccination_data, on= 'date')

# =============================================================================
# ICU patients
# =============================================================================

USA_icu_data= USA_data[['date', 'icu_patients']]
USA_icu_data= USA_icu_data.reset_index(drop= True)


# for column in USA_icu_data.columns:
#     print(USA_icu_data[USA_icu_data[column].isna()])
# 
# This can be used to check where NA's are, but is commented out to prevent 
# clogging up the console print

# replacing negative values with absolute values as negative values are non-sensical,
# so we assume they should simply be not negative
for i in range(len(USA_icu_data)):
    if USA_icu_data.iat[i, 1]<0:
        USA_icu_data.iat[i, 1]= np.abs(USA_icu_data.iat[i, 1])

# Start the dataset at the first non-NaN value, since we cannot know the icu patients before this point
first_icu= 175
# End the dataset at the last non-NaN value, since we cannot know the icu patients after this point
last_icu= 526
USA_icu_data= USA_icu_data.iloc[first_icu:last_icu]

# Creating normalised case variables (per million for easy interpretation)
# Total deaths per hundred for easy interpretation
total_icu_per_million= [value/population_USA*1000000 for value in USA_icu_data['icu_patients']]
USA_icu_data['total_icu_per_million']= total_icu_per_million

USA_icu_data= USA_icu_data.merge(USA_vaccination_data, on= 'date')



FRA_icu_data= FRA_data[['date', 'icu_patients']]
FRA_icu_data= FRA_icu_data.reset_index(drop= True)


# for column in FRA_icu_data.columns:
#     print(FRA_icu_data[FRA_icu_data[column].isna()])
#
# This can be used to check where NA's are, but is commented out to prevent 
# clogging up the console print

# replacing negative values with absolute values as negative values are non-sensical,
# so we assume they should simply be not negative
for i in range(len(FRA_icu_data)):
    if FRA_icu_data.iat[i, 1]<0:
        FRA_icu_data.iat[i, 1]= np.abs(FRA_icu_data.iat[i, 1])

# Start the dataset at the first non-NaN value, since we cannot know the icu patients before this point
first_icu= 53
# End the dataset at the last non-NaN value, since we cannot know the icu patients after this point
last_icu= 521
FRA_icu_data= FRA_icu_data.iloc[first_icu:last_icu]

# Creating normalised case variables (per million for easy interpretation)
# Total deaths per hundred for easy interpretation
total_icu_per_million= [value/population_FRA*1000000 for value in FRA_icu_data['icu_patients']]
FRA_icu_data['total_icu_per_million']= total_icu_per_million

FRA_icu_data= FRA_icu_data.merge(FRA_vaccination_data, on= 'date')
# =============================================================================
# Reproduction rate
# =============================================================================

USA_R_data= USA_data[['date', 'reproduction_rate']]
USA_R_data= USA_R_data.reset_index(drop= True)


# for column in USA_R_data.columns:
#     print(USA_R_data[USA_R_data[column].isna()])
#
# This can be used to check where NA's are, but is commented out to prevent 
# clogging up the console print

# replacing negative values with absolute values as negative values are non-sensical,
# so we assume they should simply be not negative
for i in range(len(USA_R_data)):
    if USA_R_data.iat[i, 1]<0:
        USA_R_data.iat[i, 1]= np.abs(USA_R_data.iat[i, 1])

# Start the dataset at the first non-NaN value, since we cannot know the icu patients before this point
first_R= 43
# End the dataset at the last non-NaN value, since we cannot know the icu patients after this point
last_R= 526
USA_R_data= USA_R_data.iloc[first_R:last_R]

USA_R_data= USA_R_data.merge(USA_vaccination_data, on= 'date')


FRA_R_data= FRA_data[['date', 'reproduction_rate']]
FRA_R_data= FRA_R_data.reset_index(drop= True)


# for column in FRA_R_data.columns:
#     print(FRA_R_data[FRA_R_data[column].isna()])
#
# This can be used to check where NA's are, but is commented out to prevent 
# clogging up the console print

# replacing negative values with absolute values as negative values are non-sensical,
# so we assume they should simply be not negative
for i in range(len(FRA_R_data)):
    if FRA_R_data.iat[i, 1]<0:
        FRA_R_data.iat[i, 1]= np.abs(FRA_R_data.iat[i, 1])

# Start the dataset at the first non-NaN value, since we cannot know the icu patients before this point
first_R= 37
# End the dataset at the last non-NaN value, since we cannot know the icu patients after this point
last_R= 524
FRA_R_data= FRA_R_data.iloc[first_R:last_R]

FRA_R_data= FRA_R_data.merge(FRA_vaccination_data, on= 'date')
