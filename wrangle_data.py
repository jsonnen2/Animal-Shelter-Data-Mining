
import os
import pandas as pd
import numpy as np

def preprocess_for_sklearn_tree(X_train: pd.DataFrame, y_train, X_val: pd.DataFrame, y_val, 
                                X_test: pd.DataFrame, y_test, cont_mask):
    '''
    Trims train and val X sets to have all the same categories
        np.unique(X_train) == np.unique(X_val) == np.unique(X_test)
    Converts data to a onehot encoding
    '''

    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(sparse_output=False)

    for col in cont_mask.index:
        if cont_mask[col] == 0:
            
            valid_values = np.intersect1d(
                np.intersect1d(X_train[col].unique(), X_val[col].unique()),
                X_test[col].unique()
            )
            # trim train set to categories that exist in all sets
            y_train = y_train[X_train[col].isin(valid_values)]
            X_train = X_train[X_train[col].isin(valid_values)]
            # trim train set to categories that exist in all sets
            y_val = y_val[X_val[col].isin(valid_values)]
            X_val = X_val[X_val[col].isin(valid_values)]
            # trim train set to categories that exist in all sets
            y_test = y_test[X_test[col].isin(valid_values)]
            X_test = X_test[X_test[col].isin(valid_values)]


    # One hot encode all categorical features
    onehot_xtrain = []
    onehot_xval = []
    onehot_xtest = []

    for col in cont_mask.index:  # Iterate by column index
        if cont_mask[col] == 0:
            # Fit and transform categorical data using OneHotEncoder
            train_col = enc.fit_transform(X_train[col].to_numpy().reshape(-1, 1))
            onehot_xtrain.append(train_col)  # Convert sparse matrix to dense
            val_col = enc.transform(X_val[col].to_numpy().reshape(-1, 1))
            onehot_xval.append(val_col)
            test_col = enc.transform(X_test[col].to_numpy().reshape(-1, 1))
            onehot_xtest.append(test_col)
        else:
            # Append continuous data directly
            onehot_xtrain.append(X_train[col].to_numpy().reshape(-1, 1))
            onehot_xval.append(X_val[col].to_numpy().reshape(-1, 1))
            onehot_xtest.append(X_test[col].to_numpy().reshape(-1, 1))

    # Concatenate the columns to form the final arrays
    X_train = np.concatenate(onehot_xtrain, axis=1)
    X_val = np.concatenate(onehot_xval, axis=1)
    X_test = np.concatenate(onehot_xtest, axis=1)

    return X_train, y_train, X_val, y_val, X_test, y_test

# select a target
# onehot categorical variables (have user make a bitmask for each feature)

# split into train/test

def my_print(data):
    col_width = max(len(str(row[0])) for row in data)
    for row in data:
        print(f"{row[0]:<{col_width}}  {row[1]}")

if __name__ == '__main__':
    # fix printing of numpy arrays
    np.set_printoptions(linewidth=os.get_terminal_size().columns)

    # load data
    data = pd.read_csv("data/raw_data.csv")

    # drop 1 value columns 
    data = data.drop(columns=["animal_type", "count"])
    
    # Remove all age and time features. 
    # The only time I use is outcome_age_(days) [categorical with 44 bins]
    data = data.drop(columns=["date_of_birth", "datetime", "monthyear", "outcome_month", "outcome_year", "outcome_weekday", "outcome_hour",
                              "dob_year", "dob_month", "dob_monthyear", "age_upon_outcome", "outcome_age_(years)", "sex_age_outcome"])
    # rename some of the columns I kept
    data = data.rename(columns={"Period Range": "period_range", "Spay/Neuter":"spay/neuter", "outcome_age_(days)": "outcome_age",
                        "Cat/Kitten (outcome)": "cat/kitten"})

    # color1 and color2 represent the color feature well. I don't need color and nan in color2 just means the cat has only 1 color
    data = data.drop(columns=['color'])
    data['color2'] = data['color2'].map(lambda x: "none" if pd.isna(x) else x)
    # nearly all (99.8%) of cats have 1 breed. So I just use the breed1 feature.
    data = data.drop(columns=["breed", "breed2"])
    # handle nan values
    data = data.dropna(subset=["outcome_type"])
    data['coat_pattern'] = data['coat_pattern'].map(lambda x: "missing" if pd.isna(x) else x)
    
    # change name to be binary: does animal have a name?
    data['name'] = data['name'].map(lambda x: 0 if pd.isna(x) else 1)
    
    # categorize outcome_age into bins:
    bins = [0, 7, 30, 180, 330, 1095, 2190, 3650, float('inf')]
    labels = ['0-7 days', '14-30 days', '1-6 monthes', '6-12 monthes', '1-3 years', '4-6 years', '7-10 years', '>10 years']
    data['outcome_age'] = pd.cut(data['outcome_age'], bins=bins, labels=np.arange(len(labels)), right=True, include_lowest=True)

    # replace nan in outcome_subtype with an empty string ""
    data['outcome_subtype'] = data['outcome_subtype'].map(lambda x: "" if pd.isna(x) else x)
    # string concate outcome_type with outcome_subtype
    data['outcome'] = data['outcome_type'] + ' ' + data['outcome_subtype']
    data = data.drop(columns=['outcome_type', 'outcome_subtype'])

    # onehot categorical features
    categorical_names = ["breed", "color", "outcome_subtype", "outcome_type", "sex_upon_outcome", "sex", "spay/neuter", "Periods",
                         "period_range", "cat/kitten", "age_group", "breed1", "breed2", "cfa_breed", "domestic_breed", "coat_pattern",
                         "color1", "color2", "coat"]
    # TODO

    # result should be a numpy matrix of ints
    data.to_csv("data/processed_data.csv", index=False)
