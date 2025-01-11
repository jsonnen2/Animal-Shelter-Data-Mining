# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
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
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def bootstrap(features: pd.array, targets: np.array, size):
    '''
    sampling with replacement in the range (0, n) I will expect n(1 - (1 - 1/n)^k ) unique datapoints. 
    k = n:  n(1 - e^-1) = 0.632n
    k = 2n: n(1 - e^-2) = 0.865n
    k = 3n: n(1 - e^-3) = 0.950n --> 38,000 (30,500 train; 7500 val)
    k = 4n: n(1 - e^-4) = 0.982n
    k = 5n: n(1 - e^-5) = 0.993n --> 5120 (4096 train; 1024 val)

    Return
    ------
    x_train -- pd.ndarray
    x_val   -- pd.ndarray
    y_train -- np.ndarray
    y_val   -- np.ndarray
    '''
    n = len(features)
    all_indices = np.arange(n)
    discard_indices = np.random.choice(all_indices, size=size, replace=True)
    usable_indices = np.setdiff1d(all_indices, discard_indices)
    np.random.shuffle(usable_indices)

    # 80% train, 20% val split
    split = int(0.8 * len(usable_indices))
    train_idx = usable_indices[:split]
    val_idx = usable_indices[split:]
    
    # x_train, y_train, x_val, y_val
    return features.iloc[train_idx], targets[train_idx], features.iloc[val_idx], targets[val_idx]


if __name__ == '__main__':
    classifier_names = [
        "Majority Classifier",
        "Nearest Neighbors",
        # "Linear SVM",
        # "RBF SVM",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]

    classifiers = [
        DummyClassifier(strategy="most_frequent"),
        KNeighborsClassifier(3),
        # SVC(kernel="linear", C=0.025),
        # SVC(kernel="rbf", gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(n_estimators=10, max_features=1, max_depth=5, bootstrap=False),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(algorithm="SAMME"),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]
    # Hyperparameters
    bootstrap_factor = 3
    bootstrap_trials = 10

    # load datasets as numpy vectors
    # TODO: what form does the dataset need to take?
    training_set = np.loadtxt("data/train/sklearn_ready.csv")
    testing_set = np.loadtxt("data/test/sklearn_ready.csv")

    dataset_names = [
        "Everything"
    ]
    datasets = [
        (training_set[:-1], testing_set[:-1], training_set[-1], testing_set[-1])
    ]

    # Store accuracies for the trials
    output = pd.DataFrame(columns=np.arange(bootstrap_trials), index=classifier_names)

    for classifier_name, classifier in zip(classifier_names, classifiers):
        for ds_name, ds in zip(dataset_names, datasets):
            for idx in range(bootstrap_trials):

                # split into training and test part
                X, X_test, y, y_test, cont_mask = ds 

                # perform bootstrapping
                n = len(X)
                X_train, y_train, X_val, y_val = bootstrap(X, y, n*bootstrap_factor)
                
                print("fitting")
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

