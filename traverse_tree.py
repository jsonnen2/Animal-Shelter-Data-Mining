import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder


# load datasets as numpy vectors
training_set = pd.read_csv("data/train.csv").to_numpy()[:,:-1] # remove 2nd target variable
testing_set = pd.read_csv("data/test.csv").to_numpy()[:,:-1]

num_classes = len(np.unique(training_set[:,-1]))
y = training_set[:,-1]
y_test = testing_set[:,-1]

# onehot encode
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
training_set = encoder.fit_transform(training_set)
testing_set = encoder.transform(testing_set)

X = training_set[:,:-num_classes]
# y = training_set[:,-num_classes:]
X_test = testing_set[:,:-num_classes]
# y_test = testing_set[:,-num_classes:]



# Load example data
data = load_iris()
X, y = data.data, data.target
print(X)
# Fit a decision tree
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)

# Access the tree structure
tree = clf.tree_

# Initialize a list to store decision rules
decision_rules = []

def traverse_tree(node_id=0, depth=0):
    """Recursive function to traverse the tree."""
    # If it's not a leaf node
    if tree.feature[node_id] != -2:
        # Get the feature index and threshold value
        print(tree.feature)
        feature_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        
        # Store the feature index, threshold, and depth
        decision_rules.append({
            "feature": feature_idx,
            "threshold": threshold,
            "depth": depth
        })
        
        # Traverse the left and right children
        traverse_tree(tree.children_left[node_id], depth + 1)
        traverse_tree(tree.children_right[node_id], depth + 1)

# Start traversing from the root node
traverse_tree()

# Display the decision rules
for rule in decision_rules:
    print(f"Depth: {rule['depth']}, Feature: {rule['feature']}, Threshold: {rule['threshold']}")
