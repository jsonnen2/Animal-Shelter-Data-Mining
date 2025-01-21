
import os
import pandas as pd
import numpy as np


def check_for_nan(data):
    # Iterates through all columns and counts how many times nan appears
    for col in data.columns:
        count_of_nan = pd.isna(data[col]).sum()
        print(f"{col} --> {count_of_nan}")

if __name__ == '__main__':
    # fix printing of numpy arrays
    np.set_printoptions(linewidth=os.get_terminal_size().columns)

    # load data
    data = pd.read_csv("data/Austin_Animal_Center_Outcomes.csv")

    # drop unneccessary columns
    data = data.drop(columns=["Animal ID", "MonthYear"])
    
    # rename some of the columns I kept
    data = data.rename(columns={"DateTime": "data_entry_time", "Date of Birth": "DOB"})

    # often, there is no subtype for outcome_type. Solution: replace nan with "none"
    data['Outcome Subtype'] = data['Outcome Subtype'].map(lambda x: "none" if pd.isna(x) else x)

    # These columns have very few nans. So I just remove every instance of nan
    data = data[data["Outcome Type"].notna()] # 43 nan
    data = data[data["Sex upon Outcome"].notna()] # 2 nan
    data = data[data["Age upon Outcome"].notna()] # 6 nan
    
    # change name to be binary: does animal have a name?
    data['Name'] = data['Name'].map(lambda x: 0 if pd.isna(x) else 1)

    # string parse the hour at time of outcome. 
    hours = data['data_entry_time'].str.split(" ").str[1].str.split(":").str[0].astype(int)
    am_pm = data['data_entry_time'].str.split(" ").str[2]
    hours = hours + (am_pm == "PM") * 12 # Add 12 when the time is PM.
    data['hour_of_outcome'] = hours % 24  # Handle 12 AM properly

    # string parse the year at time of outcome
    data['year_of_outcome'] = data['data_entry_time'].str.split(" ").str[0].str.split("/").str[2].astype(int)

    # compute the day of week at time of outcome.
    datetime_data = pd.to_datetime(data['data_entry_time'], format="%m/%d/%Y %I:%M:%S %p")
    data['day_of_week_at_outcome'] = datetime_data.dt.day_name()

    # reorder features so the targets are last
    feature_names = ['Name', 'data_entry_time', 'DOB', 'hour_of_outcome', 'day_of_week_at_outcome', 
       'Animal Type', 'Sex upon Outcome', 'Age upon Outcome', 'Breed', 'Color',
       'Outcome Type', 'Outcome Subtype',
       ]
    data = data[feature_names]

    data = data.drop(columns=["data_entry_time", "DOB", "hour_of_outcome"])

    data.iloc[:,:-1].to_csv("data/processed_data.csv", index=False)

    # train/test split
    n = len(data)
    np.random.seed(42)
    index = np.random.permutation(np.arange(n))
    split = int(np.floor(0.9*n))
    train = data.iloc[index[:split]]
    test = data.iloc[index[split:]]

    # save as a pandas dataframe
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)
