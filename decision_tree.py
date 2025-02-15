import numpy as np
import pandas as pd
import sys

from scipy.stats import entropy
from sklearn.feature_selection import chi2
from sklearn.preprocessing import OneHotEncoder


def split_by_random(x, y, unseen):
    return np.random.choice(np.where(unseen)[0])

def split_by_entropy(x, y, unseen):
    # search for decision rule with minimum entropy
    min_entropy = np.Inf
    decision_split = 0

    # iterate over decision rules in a random order for 
    # variation when there are ties in minimum entropy
    rand_order = np.random.permutation(np.where(unseen)[0])
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

    return decision_split

def split_by_chi(x, y, unseen):
    # compress y from (n_samples, n_classes) into (n_samples,)
    y = y @ np.arange(y.shape[1])
    chi_stat, p_val = chi2(x[:, unseen], y)
    # choose decision split as the maximum chi stat
    # TODO: use p-value to terminate tree
    return np.argmax(chi_stat)

class Node():

    def __init__(self, 
                rule: int,
                majority_class: int,
                true_child: 'Node',
                false_child: 'Node'):

        self.rule = rule # column # of feature
        self.majority_class = majority_class # stores majority class prediction as column #
        self.true_child = true_child # true side of rule
        self.false_child = false_child # false side of rule

class Decision_Tree():
    def __init__(self, split_type="random", max_features=np.Inf, max_depth = np.Inf, epsilon=0.0):
        '''
        Parameters
        ----------
        split_type : str
            The type of data splitting to be used, by default "random".
            Options include:
            - "random" : split the data randomly.
            - "entropy" : split the data by minimizing entropy.
            - "chi2" : split the data by maximizing chi-squared statistic
        max_features : int
            The decision tree will select max_features many features to consider at each step of the decision tree.
        max_depth : int
            The decision tree will select max_features many features to consider at each step of the decision tree.
        epsilon : float
            If entropy is below this value, predict the majority class.
        '''
        self.epsilon = epsilon
        self.split_type = split_type
        self.max_features = max_features
        self.max_depth = max_depth

        # statistics about my tree
        self.num_nodes = 0
        self.num_leaves = 0


    def fit(self, x: np.array, y: np.array, unseen=None, depth=0):
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
        if x.shape[0] == 0: # empty data
            return None

        if unseen is None: # initialize a list of features which are not traversed
            unseen = [True] * x.shape[1]

        # find majority class of data
        unique, counts = np.unique(y @ np.arange(y.shape[1]), return_counts=True)
        majority_class = unique[np.argmax(counts)]

        # create node
        node = Node(None, majority_class, None, None)

        # check if the tree terminates here
        if depth >= self.max_depth:
            return node
        
        # find decision split
        criterion = {
            "random" : split_by_random,
            "entropy" : split_by_entropy,
            "chi2" : split_by_chi,
        }
        # select a decision split according entroy, chi2, or random.
        split_function = criterion.get(self.split_type)
        node.decision_split = split_function(x, y, unseen)

        # update tracking mask
        unseen[node.decision_split] = False

        true_idx = x[:, node.decision_split] == 1
        node.true_child = self.fit(x[true_idx], y[true_idx], unseen, depth+1)
        node.false_child = self.fit(x[~true_idx], y[~true_idx], unseen, depth+1)

        return node


    def traverse_tree(self, features, targets, node: 'Node'):
        """
        Finds the depth in the tree for each decision rule
        Return {rule column : depth in tree}
        """
        
        # TODO: interprete node.rule & node.majority_class to create a formatted rule
    
    # TODO: column no. to formatted rule method
    def print_tree(self, tree: 'Node', file = sys.stdout, indent="\n"):
        """
        Prints a visual representation of the decision tree to the specified file.
        Use sys.stdout to print to console.

        Parameters
        ----------
        tree : Node
            The decision tree model to be printed.
        file : str
            The filepath specified as a string
        indent : str, optional
            DO NOT INITIALIZE
            Use to add indenting for each branch. Lower level branches have more indentation.

        Returns
        -------
        None
            This method prints the tree to the specified file and does not return a value.
        """
        indent = indent + "."

        # leaf node
        if tree.true_child == None and tree.false_child == None:
            print("\t leaf node. class=" + str(tree.majority_class), file=file, end="")
            return
        
        # print node
        print(indent + str(tree.decision_split) + " " + str(tree.majority_class), file=file, end="")
        # recurse on children
        if tree.true_child != None:
            self.print_tree(tree.true_child, file, indent + ".")
        if tree.false_child != None:
            self.print_tree(tree.false_child, file, indent + ".")

    
data = pd.read_csv("data/test.csv").to_numpy()[:,:-1]
num_classes = len(np.unique(data[:,-1]))
encoder = OneHotEncoder(sparse_output=False)
data = encoder.fit_transform(data)
x = data[:,:-num_classes]
y = data[:,-num_classes:]

tree = Decision_Tree(max_depth=10, split_type="chi2")
root = tree.fit(x, y)
tree.print_tree(root)