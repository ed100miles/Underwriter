import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np

# Import and convert data to df:
datapath = Path('./data/car_insurance_claim.csv')
df = pd.read_csv(datapath)

# Clean Dollars data:

def make_int64_and_nans_to_mean(df, *args:str):
    """ Takes df obj and strings of fields to  """
    df_to_transform = df
    for field in args:
        df_to_transform[field] = df_to_transform[field].fillna(df_to_transform[field].mean())
        df_to_transform[field] = df_to_transform[field].astype('int64')
    return df_to_transform


def dollars_to_int64s_and_nans_to_mean(df, *args:str):
    '''removes '$' and ',' from dollar values to convert to int. NANs to mean.'''
    for dollar_field in args:
        for index, entry in enumerate(df[dollar_field]):
            if type(entry) == str:
                dollar_val = int(entry[1:].replace(',', ''))
                df.at[index, dollar_field] = dollar_val
        make_int64_and_nans_to_mean(df, dollar_field)
    return df
    
clean_df = dollars_to_int64s_and_nans_to_mean(df, 'INCOME', 'HOME_VAL', 'BLUEBOOK', 'OLDCLAIM', 'CLM_AMT')

clean_df = make_int64_and_nans_to_mean(clean_df, 'AGE', 'CAR_AGE')


# print(clean_df.info())
# print(clean_df.head())
