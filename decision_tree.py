import numpy as np
import pandas as pd

from scipy.stats import entropy
from sklearn.feature_selection import chi2
from sklearn.preprocessing import OneHotEncoder


def split_by_random(x, y):
    n = x.shape[1]
    return np.random.randint(1, n)

def split_by_entropy(x, y):
    # search for decision rule with minimum entropy
    min_entropy = np.Inf

    # iterate over decision rules in a random order for 
    # variation when there are ties in minimum entropy
    rand_order = np.random.permutation(np.arange(x.shape[1]))
    for i in rand_order:
        vect = x[:,i]
        # grab indices where decision rule is true
        index = vect == 1
        # find the probability for each outcome where the decision rule is true
        probabilities = np.sum(y[index], axis=0) / np.sum(index)
        # calculate entropy using a scipy package
        e = entropy(probabilities, base=2)
        
        if e < min_entropy:
            min_entropy = e
            decision_split = i

    return decision_split, min_entropy

def split_by_chi(x, y):
    # convert y from (n_samples, n_classes) into (n_samples,)
    # essentially a conversion from onehot to integer-hot
    y = y @ np.arange(y.shape[1])
    return chi2(x, y)

class Node():

    def __init__(self, 
                rule: int,
                true_child: 'Node',
                false_child: 'Node',
                majority_class: int):

        self.rule = rule # column # of feature
        self.true_child = true_child # true side of rule
        self.false_child = false_child # false side of rule
        self.majority_class = majority_class # stores majority class prediction as column #

class Decision_Tree():
    def __init__(self, split_type="random", max_features=np.Inf, max_depth = np.Inf, epsilon=0.0):
        '''
        Parameters
        ----------
        split_type : str
            The type of data splitting to be used, by default "random".
            Options include:
            - "in_order" : split the data in order
            - "random" : split the data randomly.
            - "entropy" : split the data by minimizing entropy.
            - "chi2" : split the data by maximizing chi-squared statistic
            - "gain_ratio" : split the data by maximizing gain-ratio (a variant of entropy)
        max_features : int
            The decision tree will select max_features many features to consider at each step of the decision tree.
        max_depth : int
            The decision tree will select max_features many features to consider at each step of the decision tree.
        epsilon : float
            If entropy is below this value, predict the majority class.
            the subtree is "sufficiently pure"
        '''
        self.epsilon = epsilon
        self.split_type = split_type
        self.max_features = max_features
        self.max_depth = max_depth

        # statistics about my tree
        self.num_nodes = 0
        self.num_leaves = 0


    def fit(self, x: np.array, y: np.array):
        """
        Creates a Decision Tree with the TDIDT algorithm using the provided features and targets.

        Parameters
        ----------
        features : np.array
            It is assumed onehotting has been performed and all features are categorical.
            Continuous data can be binned using KBinsDiscretizer
        targets : np.array
            The target values for each sample in the training data. Onehotted and categorical.
        """

        # find decision split
        criterion = {
            "random" : split_by_random,
            "entropy" : split_by_entropy,
            "chi2" : split_by_chi,
        }
        # 2. select a attribute of features to split on
        split_function = criterion.get(self.split_type)
        split_column = split_function(x, y)


    def traverse_tree(self, features, targets, node: 'Node'):
        """
        Finds the depth in the tree for each decision rule
        Return {formatted rule : depth in tree}
        """
        
        # TODO: interprete node.rule & node.majority_class to create a formatted rule
    
data = pd.read_csv("data/test.csv").to_numpy()[:,:-1]
num_classes = len(np.unique(data[:,-1]))
encoder = OneHotEncoder(sparse_output=False)
data = encoder.fit_transform(data)
x = data[:,:-num_classes]
y = data[:,-num_classes:]

chi, vals = split_by_chi(x,y)
print(np.max(vals))