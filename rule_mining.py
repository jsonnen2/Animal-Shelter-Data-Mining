
# Core idea is to mine for association rules

import pandas as pd
import numpy as np
from itertools import product


data = pd.read_csv("data/train.csv")
features = data.columns[:-2]
targets = data.columns[-2:] # TODO: change to have outcome_type and outcome_subtype seperate. len(targets)=2
storage = {}

for column in features:
    unique, counts = np.unique(data[column].to_numpy(), return_counts=True)
    storage[column] = (unique, counts)

# Consider only atomic rules
# for column in features: 
#     for rule in data[column].unique():
#         subset = data[column] == rule
#         target_names, distribution_of_targets = np.unique(data[targets[0]][subset].to_numpy(), return_counts=True)
#         confidence = distribution_of_targets / subset.sum()
        
#         majority_class = target_names[np.argmax(distribution_of_targets)]
#         print(f"{column} = {rule} --> {majority_class}")
#     print("==========")



# method written by claude.ai
def get_value_combinations(data, selected_features):
    unique_values = {feat: data[feat].unique() for feat in selected_features}
    feature_values = [unique_values[feat] for feat in selected_features]
    for combination in product(*feature_values):
        yield [*zip(selected_features, combination)]


rule = []
# for i in range(1, 2**len(features)): # this considers ALL combinations of features
#     binary = np.array([int(b) for b in f'{i:0{len(features)}b}'])
for i in range(len(features)):  # this considers only ATOMIC features
    binary = np.array([1 if j == i else 0 for j in range(len(features))])
    print(binary)
    names = features[binary.astype(bool)]
    for antecedent in get_value_combinations(data, names):
        
        # for j in range(1, 2**len(targets)): # iterate through ALL possible targets
            # binary = np.array([int(a) for a in f'{j:0{len(targets)}b}'])
        for j in range(len(targets)): # iterate through only ATOMIC targets
            binary = np.array([1 if k == j else 0 for k in range(len(targets))])

            names = targets[binary.astype(bool)]
            for predicate in get_value_combinations(data, names):
                rule.append( (antecedent, predicate) )

for r in rule:
    print(r)