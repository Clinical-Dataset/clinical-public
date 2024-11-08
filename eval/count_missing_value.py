import pandas as pd
import numpy as np
import pickle
import json

# Load data
train_df = pd.read_csv('../../data_llm/enrollment_timefiltered_train.csv', sep='\t')
test_df = pd.read_csv('../../data_llm/enrollment_timefiltered_test.csv', sep='\t')

# Concatenate training and testing DataFrames
trial_df = pd.concat([train_df, test_df], sort=False)

# Check for missing values that are null
missing_null = trial_df.isnull()

# Check for missing values that are -1
missing_neg_one = (trial_df == -1)

# Combine the two conditions to identify all missing values
missing_combined = missing_null | missing_neg_one

# Output the results
missing_summary = missing_combined.sum()
total_missing = missing_combined.sum().sum()

print("Missing values per column:")
print(missing_summary)
print(f"\nTotal missing values: {total_missing}")

# If you want to see the rows with missing values, you can filter them:
rows_with_missing = trial_df[missing_combined.any(axis=1)]
print("\nRows with missing values:")
print(rows_with_missing)


country_dict = json.load(open('../data/country_dict.json', 'r'))
state_dict = json.load(open('../data/state_dict.json', 'r'))
city_dict = json.load(open('../data/city_dict.json', 'r'))

nctids = set(trial_df['nctid'])
country_dict = {k: v for k, v in country_dict.items() if k in nctids}
state_dict = {k: v for k, v in state_dict.items() if k in nctids}
city_dict = {k: v for k, v in city_dict.items() if k in nctids}

# count missing values for country, state, and city   nctid:[]
missing_country = 0
missing_state = 0
missing_city = 0
for nctid in nctids:
    if country_dict[nctid] == []:
        missing_country += 1
    if state_dict[nctid] == []:
        missing_state += 1
    if city_dict[nctid] == []:
        missing_city += 1

    
print(f"Missing country: {missing_country}")
print(f"Missing state: {missing_state}")
print(f"Missing city: {missing_city}")

    


