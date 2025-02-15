
# Core idea is extract a set of if then rules that are most descriptive
# SVD as a technique to find most descriptive orthogonal vector set

import pandas as pd
import numpy as np
from itertools import product
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import entropy


# method written by claude.ai
def get_value_combinations(data, selected_features):
    unique_values = {feat: data[feat].unique() for feat in selected_features}
    feature_values = [unique_values[feat] for feat in selected_features]
    for combination in product(*feature_values):
        yield [*zip(selected_features, combination)]

def all_combos(n: int):
    # TODO: change counting order. one one, two one, three one, ...
    '''
    Yields a binary number counting from 0 to 2**n
    n: int -- number of bits for binary array
    '''
    for i in range(1, 2**n): 
        yield np.array([int(b) for b in f'{i:0{n}b}'])

def atomic_combos(n: int):
    '''
    Yields a binary number counting with exactly one 1.
    n: int -- number of bits for binary array
    '''
    for i in range(n):
        yield np.array([1 if j == i else 0 for j in range(n)])


def find_subset(data, antecedent, onehot_features, feature_header):

    # Find the subset of data which follows the antecedent rule
    subset_index: list[bool] = np.ones(len(data), dtype=bool)
    for attribute in antecedent:
        name = f"{attribute[0]}_{attribute[1]}"
        index = np.where(feature_header == name)[0][0]
        bitmask = onehot_features[:, index].astype(bool)
        subset_index = np.logical_and(subset_index, bitmask)
    return data[subset_index]

def majority_class(targets: pd.DataFrame):
    '''
    find the class distribution as a probability

    targets: pd.DataFrame
        targets for the entire dataset or a subset which follows some antecedent
    '''
    return targets.mode().iloc[0].to_numpy()


def entropy_in_antecedent(targets: pd.DataFrame):
    '''
    Entropy is calculated across target classes. 

    Return: list[int]
        The entropy for each column of targets
    '''
    n = len(targets.columns)
    shannon_entropy = np.zeros(n)
    for i in range(0, n):
        unique, counts = np.unique(targets.iloc[:, i].to_numpy(), return_counts=True)
        shannon_entropy[i] = entropy(counts)
    return shannon_entropy


def chi2_in_antecedent(antecedent):
    '''
    compute chi2 p-values for the antecedent of a rule.

    '''
    from sklearn.feature_selection import chi2
    chi_stat, p_val = chi2(x[:, unseen], y)

def noguchi_tree():
    '''
    1. fit 1000 decision trees to independent bootstrap samples
    2. Feed a sample (antecedent with Any for unspecified)
       Return probability of each class for each tree.
    3. Average over 1000 trees
    '''

def monte_carlo_estimate(rule):
    '''
    Use a resampling method to generate Monte Carlo estimate for mean and variance
    Type I error rate = P(reject H0 | H0 is true)
        - assumes normal distribution

    1. Generate B m-sized resamples from the training dataset
    2. Use logistic regression to estimate probability of target class
       Alternatively, find distribution over targets for resample
    '''

if __name__ == "__main__":
    data = pd.read_csv("data/train.csv")
    features: list[str] = data.columns[:-2]
    targets: list[str] = data.columns[-2:]
    rule = []
    all_entropy = []
    length = []
    majority_predictor = []

    # onehot the data
    # create mapping from (feature, value) to column number
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    onehot_features = encoder.fit_transform(data[features])
    feature_header: list[str] = encoder.get_feature_names_out(features)
    onehot_targets = encoder.fit_transform(data[targets])
    target_header: list[str] = encoder.get_feature_names_out(targets)

    print(data[features].nunique())

    # iterate a binary sequence of features
    for binary in atomic_combos(len(features)):
        print(binary)
        names = features[binary.astype(bool)]

        # iterate values assigned to these features
        for i, antecedent in enumerate(get_value_combinations(data, names)):
            if i % 1000 == 0:
                print(i)
            data_subset = find_subset(data, antecedent, onehot_features, feature_header)
            if len(data_subset) < 50:
                continue

            rule.append(antecedent)

            # find a statistic for this data
            shannon_entropy = entropy_in_antecedent(data_subset[targets])
            all_entropy.append(shannon_entropy)

            length.append(len(data_subset))

            mode = majority_class(data_subset[targets])
            majority_predictor.append(mode)


        df = pd.DataFrame(zip(rule, all_entropy, length, majority_predictor), columns=["Rule", "Entropy across targets", "Length Following Antecedent", "Majority Class"])
        df.to_csv("data/limited_all_combo_entropy_search.csv", index=False)

    all_entropy = np.vstack(all_entropy)
    majority_predictor = np.vstack(majority_predictor)

    # find minimum entropy
    sorted_idx = np.argsort(all_entropy[:,0])

    for i in sorted_idx[:100]:
        print("=============")
        print(rule[i])
        print(all_entropy[i])
        print(majority_predictor[i])

