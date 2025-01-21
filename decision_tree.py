import numpy as np

from scipy.stats import entropy
from sklearn.feature_selection import chi2

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
    def __init__(self, split_type="random", max_features=np.Inf, max_depth = np.Inf,
                epsilon=0.0, continuous_mask = None):
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


    def fit(self, x: np.array, y: np.array, continuous_mask = None):
        """
        Creates a Decision Tree with the TDIDT algorithm using the provided features and targets.

        Parameters
        ----------
        features : pd.DataFrame
            The input features for training, with each row representing a sample and each column a feature.
        targets : pd.Series
            The target values for each sample in the training data.
        continuous_mask : list[bool]
            A bitmask declaring which features are continuous (1) and which are categorical (0)
        """



    def traverse_tree(self, features, targets, node: 'Node'):
        """
        Finds the depth in the tree for each decision rule
        Return {formatted rule : depth in tree}
        """
        
        # TODO: interprete node.rule & node.majority_class to create a formatted rule
    