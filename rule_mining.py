
# Core idea is to mine for association rules

import pandas as pd
import numpy as np
from itertools import product


data = pd.read_csv("data/train.csv")
features = data.iloc[:,:-1]
targets = data.iloc[:,-1]
storage = {}

for col in features.columns:
    unique, counts = np.unique(features[col].to_numpy(), return_counts=True)
    storage[col] = (unique, counts)

# Consider only atomic rules
for col in features.columns: 
    for rule in features[col].unique():
        subset = features[col] == rule
        target_names, distribution_of_targets = np.unique(targets[subset].to_numpy(), return_counts=True)
        confidence = distribution_of_targets / subset.sum()
        
        majority_class = target_names[np.argmax(distribution_of_targets)]
        print(f"{col} = {rule} --> {majority_class}")
    print("==========")



# # method written by claude.ai
# def get_value_combinations(storage, names):
#     # Get all unique values for each selected feature
#     value_lists = [storage[name][0] for name in names]
    
#     # Use itertools.product to get all combinations
#     for values in product(*value_lists):
#         combination = dict(zip(names, values))
#         yield combination

# # Consider all rules which point to target
# n_bits = len(features.columns)
# for i in range(2**n_bits): # count in binary. this considers all combinations of features in the antecedent.
#     binary = np.array([int(b) for b in f'{i:0{n_bits}b}'])
#     names = features.columns[binary.astype(bool)]
#     for rule in get_value_combinations(storage, names):
#         # find the number of datapoints following rule
#         # and the distribution of their targets

#         subset = [] # O(n) algorithm which finds index of datapoints following the rule
#         target_names, distribution_of_class = np.unique(targets.iloc[subset].to_numpy(), return_counts=True)
#         confidence = distribution_of_class / len(subset)
#         print(rule)
#         print(target_names[np.argmax(confidence)])