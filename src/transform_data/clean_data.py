from numpy import dtype, int64, uint8
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Import and convert data to df:
datapath = Path('./data/car_insurance_claim.csv')
df = pd.read_csv(datapath)

# Clean functions:

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

def fill_nans_with_mode(df, *args:str):
    for arg in args:
        df[arg] = df[arg].fillna('Unknown')
    return df


clean_df = dollars_to_int64s_and_nans_to_mean(df, 'INCOME', 'HOME_VAL', 'BLUEBOOK', 'OLDCLAIM', 'CLM_AMT')
clean_df = make_int64_and_nans_to_mean(clean_df, 'AGE', 'CAR_AGE', 'YOJ')
"""
There are 600+ nans in 'OCCUPATION' but all have an income so not unemployed,
could be retired but average age doesnt support this,
the majority are educated to masters or phd, data shows BlueCollars dont tend to have this level of education
so not replacing with the mode (blue collar). Decided to create Unknown category. 
"""
clean_df['OCCUPATION'] = clean_df['OCCUPATION'].fillna('Unknown')


# Drops:

clean_df = clean_df.drop('CLM_AMT', axis=1) # this is a label so will give the game away!
clean_df = clean_df.drop('BIRTH', axis=1) # 'AGE' covers this feature sufficiently
clean_df = clean_df.drop('RED_CAR', axis=1) # No correlation between this feature and claims
clean_df = clean_df.drop('ID', axis=1) # No correlation between this feature and claims
# clean_df = clean_df.drop('HOME_PER_BLUE*', axis=1) # No correlation between this feature and claims

# clean_df = clean_df.set_index('ID')

# Deal with categorical features:
'''NOTE:Consider ordinal encoding for Education?'''

#NOTE: This works but messy!:
def cat_to_1hot(df, *args):
    for arg in args:
        df = pd.get_dummies(df, columns = [arg])
    return df

clean_df = cat_to_1hot( clean_df,
                    'EDUCATION', 
                    'OCCUPATION', 
                    'CAR_USE',
                    'CAR_TYPE')

enc = OrdinalEncoder(dtype=uint8)

def make_ordinal(df, *args):
    for arg in args:
        arr_2d = [[x] for x in df[arg]]
        df[arg] = enc.fit_transform(arr_2d)
    return df

clean_df = make_ordinal(clean_df, 'GENDER', 'PARENT1', 'MSTATUS', 'REVOKED', 'URBANICITY')



# arr = [[x] for x in clean_df['OCCUPATION']]
# arr_1hot = cat_encoder.fit_transform(arr)
# print(arr_1hot)
# clean_df['OCCUPATION'] = arr_1hot

# cat_encoder = OneHotEncoder(sparse=False)

# def cat_to_1hot(df, *args:str):
#     """Takes df object and strings of columns to replace with one-hot arrays,
#     returns modified dataframe."""
#     for arg in args:
#         arr_2d = [[x] for x in df[arg]]
#         arr_1hot = cat_encoder.fit_transform(arr_2d)
#         df[arg] = arr_1hot.astype('object')
#         print(f'arr1hot:{arr_1hot[0]}')
#     return df

# clean_df = cat_to_1hot(clean_df, 'GENDER', 'OCCUPATION')

# cat_df = clean_df[['PARENT1', 
#                     'MSTATUS', 
#                     'GENDER', 
#                     'EDUCATION', 
#                     'OCCUPATION', 
#                     'CAR_USE',
#                     'CAR_TYPE',
#                     'REVOKED',
#                     'URBANICITY']]

# cat_df_1hot = cat_encoder.fit_transform(cat_df)
# cat_df_1hot = cat_df_1hot.toarray()


if __name__ == '__main__':
    clean_df.info( )
    # print(clean_df['OCCUPATION'].head(5))
    # print(clean_df['OCCUPATION'].unique())
    # print(clean_df['OCCUPATION'].head())
    # print(clean_df.head())
    # print(clean_df['URBANICITY'].unique())
    pass






#TODO: Better method for filling NAN ages
'''
Some ages missing but entries have year of birth. Dataset appears to be from 1999 so calculating age off that.
'''

# age_nan_df = df[ df['AGE'].isnull()]

# age_nan_df = age_nan_df.astype({'BIRTH': 'str'})

# age_nan_df.loc[:, ('BIRTH')].apply(str)

# age_nan_df['BIRTH'].astype(str)
# # print(age_nan_df['BIRTH'].head())

# for bday in age_nan_df['BIRTH']:
#     if type(bday) != str:
#         print(bday, type(bday))
# print(age_nan_df['AGE'].head())
# print(age_nan_df)
# age_nan_df.info()
# print(age_nan_df.head())
# clean_df.describe()
# print(clean_df.info())
# print(clean_df.head())

