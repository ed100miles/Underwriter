import matplotlib.pyplot as plt
from numpy.lib.function_base import average
import pandas as pd
from pathlib import Path
import numpy as np

# Import and convert data to df:
datapath = Path('./data/car_insurance_claim.csv')
df = pd.read_csv(datapath)

# Clean Dollars data:

def make_int64_and_nans_to_mean(df, *args:str):
    for field in args:
        df[field] = df[field].fillna(df[field].mean())
        df[field] = df[field].astype('int64')

def dollars_to_int64s_and_nans_to_mean(df, *args:str):
    '''removes '$' and ',' from dollar values to convert to int. NANs unchanged.'''
    for dollar_field in args:
        for index, entry in enumerate(df[dollar_field]):
            if type(entry) == str:
                dollar_val = int(entry[1:].replace(',', ''))
                df.at[index, dollar_field] = dollar_val
        make_int64_and_nans_to_mean(df, dollar_field)
    
dollars_to_int64s_and_nans_to_mean(df, 'INCOME', 'HOME_VAL', 'BLUEBOOK', 'OLDCLAIM', 'CLM_AMT')

def get_int64_only_df(df):
    '''Returns new df object comprising of only columns with int64 datatype'''
    int_columns = []
    for column in df.columns:
        if df[column].dtype == int:
            int_columns.append(column)
    int_columns = df[int_columns]
    return int_columns.copy()

num_only_df = get_int64_only_df(df)

num_only_df.info()
df.info()

# num_only_df.hist(bins=50)

# df.hist(bins=50)
# plt.show()

# column_list = [column for column in df.columns]
# print(column_list)
# print(len(column_list))

# print(df.corr())

# plt.matshow(df.corr())
# # fig, ax = plt.subplots()

# # ax.set_xticks(df.columns)
# # ax.set_xticklabels(column for column in df.columns)

# # plt.set_xticks()
# plt.show()

