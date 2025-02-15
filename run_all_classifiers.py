import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier



def bootstrap(features: np.array, targets: np.array, bs_factor: float, train_val_split: float = 0.9):
    '''
    Bootstrap factor [0, 1]
        Represents the expected coverage of the bootstrapping

    When sampling with replacement k times, I expect the coverage (c) to be:
        1 - e^(-k/n) = c

    k = n:  (1 - e^-1) = 0.632
    k = 2n: (1 - e^-2) = 0.865
    k = 3n: (1 - e^-3) = 0.950 
    k = 4n: (1 - e^-4) = 0.982
    k = 5n: (1 - e^-5) = 0.993 

    Find the inverse:
        k = -n * ln(1 - c)
    '''
    n = features.shape[0]
    k = -n * np.log(1 - bs_factor)
    k = int(np.floor(k))

    all_indices = np.arange(n)
    # sample with replacement
    drop_idx = np.random.choice(all_indices, size=k, replace=True)
    # return indices not sampled
    keep_idx = np.setdiff1d(all_indices, drop_idx)

    split = int(np.floor(train_val_split * len(keep_idx)))
    np.random.shuffle(keep_idx)
    train_idx = keep_idx[:split]
    val_idx = keep_idx[split:]

    # x_train, y_train, x_val, y_val
    return features[train_idx], targets[train_idx], features[val_idx], targets[val_idx]


if __name__ == '__main__':
    classifier_names = [
        "Majority Classifier",
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "LDA",
    ]
    classifiers = [
        DummyClassifier(strategy="most_frequent"),
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(kernel="rbf", gamma=2, C=1),
        DecisionTreeClassifier(),
        RandomForestClassifier(bootstrap=False),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(algorithm="SAMME"),
        GaussianNB(var_smoothing=1e-9),
        LinearDiscriminantAnalysis(),
    ]
    # Hyperparameters
    bootstrap_factor = 0.85 # expected proportion of data to drop for each bootstrap trial
    bootstrap_trials = 10

    # load datasets as numpy arrays
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

    # Store accuracies for the trials
    output = pd.DataFrame(columns=np.arange(bootstrap_trials), index=classifier_names)

    for classifier_name, classifier in zip(classifier_names, classifiers):
        for idx in range(bootstrap_trials):

            # perform bootstrapping
            n = len(training_set)
            X_train, y_train, X_val, y_val = bootstrap(X, y, n*bootstrap_factor)
            
            clf = make_pipeline(StandardScaler(), classifier)
            clf.fit(X_train, y_train)
            val_score = clf.score(X_val, y_val)
            test_score = clf.score(X_test, y_test)

            print(f"{classifier_name}, {idx} ==> val: {val_score}")
            print(f"{classifier_name}, {idx} ==> test: {test_score}")
            if classifier_name not in output.index:
                output.loc[classifier_name] = [None] * len(output.columns)
            output.loc[classifier_name, idx] = str((val_score, test_score))
            output.to_csv(f"all_classifiers/Shelter_Outcome_bs={bootstrap_factor}.csv")

