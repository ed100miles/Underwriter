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
    ints_df = df[int_columns]
    return ints_df.copy()

num_only_df = get_int64_only_df(df)

# num_only_df.info()
# df.info()

# General distributions:

def show_gen_distributions():
    num_only_df.hist(bins=50)
    df.hist(bins=50)
    plt.show()
# show_gen_distributions()

'''
Create some more data?:
- Income/bluebook = value of car as % of income
- MVR / age = more points at young age suggests riskier driver
- Identify collectable cars using bluebook + age?
- OLDCLAIM / age = many claim in short time of driving suggest risk?
'''

# Create new fields:

df['BLUE%INCOME*'] = df['INCOME'] / df['BLUEBOOK']  # -0.026 - POOR
df['PTS_PER_AGE*'] = df['MVR_PTS'] / df['AGE']      # +0.232 - GREAT!
df['CLM_PER_AGE*'] = df['CLM_FREQ'] / df['AGE']      # +0.234 - GREAT!
df['RICH_KIDS*'] = df['INCOME'] / df['AGE']         # -0.112 - FAIR
df['HOME_PER_BLUE*'] = df['HOME_VAL'] / df['BLUEBOOK'] # - 0.04 - POOR
df['CLM_PER_MILE*'] = df['CLM_FREQ'] / df['TRAVTIME'] # 0.120 - FAIR
# Check Correlations:

print(df.corr()['CLAIM_FLAG'].abs().sort_values(ascending=False))

num_column_list = [column for column in df.columns if df[column].dtype == int or df[column].dtype == float]

def show_pos_neg_corrs():
    plt.matshow(df.corr(), cmap=plt.cm.RdYlBu, norm=matplotlib.colors.CenteredNorm())
    plt.xticks(range(0,len(num_column_list)), labels=num_column_list, rotation='vertical')
    plt.yticks(range(0,len(num_column_list)), labels=num_column_list, rotation='horizontal')
    plt.colorbar()
    plt.show()
# show_pos_neg_corrs()

def show_abs_corrs():
    plt.matshow(df.corr().abs(), cmap=plt.cm.Greens)
    plt.xticks(range(0,len(num_column_list)), labels=num_column_list, rotation='vertical')  
    plt.yticks(range(0,len(num_column_list)), labels=num_column_list, rotation='horizontal')
    plt.colorbar()
    plt.show()
# show_abs_corrs()

