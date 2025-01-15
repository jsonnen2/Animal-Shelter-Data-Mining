
import os
import pandas as pd
import numpy as np


if __name__ == '__main__':
    # fix printing of numpy arrays
    np.set_printoptions(linewidth=os.get_terminal_size().columns)

    # load data
    data = pd.read_csv("data/raw_data.csv")

    # drop 1 value columns 
    data = data.drop(columns=["animal_type", "count"])
    
    # Remove all age and time features. 
    # The only time I use is outcome_age_(days) [categorical with 44 bins]
    data = data.drop(columns=["date_of_birth", "datetime", "monthyear", "outcome_month", "outcome_year", "outcome_hour", "animal_id",
                              "dob_year", "dob_month", "dob_monthyear", "outcome_age_(days)", "outcome_age_(years)", "sex_age_outcome"])
    # rename some of the columns I kept
    data = data.rename(columns={"Period Range": "period_range", "Spay/Neuter":"spay/neuter", "outcome_age_(days)": "outcome_age",
                        "Cat/Kitten (outcome)": "cat/kitten"})

    # color1 and color2 represent the color feature well. I don't need color and nan in color2 just means the cat has only 1 color
    data = data.drop(columns=['color'])
    data['color2'] = data['color2'].map(lambda x: "none" if pd.isna(x) else x)
    # 99.8% of cats have 1 breed. So I drop all not-nan breed2. Then use only the breed1 feature. 
    data = data[data["breed2"].isna()]
    data = data.drop(columns=["breed", "breed2"])
    # handle nan values
    data = data.dropna(subset=["outcome_type"])
    data['coat_pattern'] = data['coat_pattern'].map(lambda x: "missing" if pd.isna(x) else x)
    
    # change name to be binary: does animal have a name?
    data['name'] = data['name'].map(lambda x: 0 if pd.isna(x) else 1)

    # replace nan in outcome_subtype with an empty string ""
    data['outcome_subtype'] = data['outcome_subtype'].map(lambda x: "" if pd.isna(x) else x)
    # string concate outcome_type with outcome_subtype
    data['outcome'] = data['outcome_type'] + '-- ' + data['outcome_subtype']
    data = data.drop(columns=['outcome_type', 'outcome_subtype'])

    # onehot categorical features 
    categorical_names = ["breed", "color", "outcome_subtype", "outcome_type", "sex_upon_outcome", "sex", "spay/neuter", "Periods",
                         "period_range", "cat/kitten", "age_group", "breed1", "breed2", "cfa_breed", "domestic_breed", "coat_pattern",
                         "color1", "color2", "coat"]
    
    # train/test split
    n = len(data)
    np.random.seed(42)
    index = np.random.permutation(np.arange(n))
    split = int(np.floor(0.9*n))
    train = data.iloc[index[:split]]
    test = data.iloc[index[split:]]

    # result should be a numpy matrix of ints
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)
