import pandas as pd
import sys
from sklearn.preprocessing import OrdinalEncoder

# Import and convert data to df:
df = pd.read_csv('./data/car_insurance_claim.csv')

# Data cleaining functions:


def features_to_dtype(df: pd.DataFrame, *args: str, to_type='int64') -> pd.DataFrame:
    """Converts columns in df to specified dype, default int64. Returns converted df."""
    for field in args:
        try:
            df[field] = df[field].astype(to_type)
        except TypeError as e:
            print(
                f'TypeError: {e}\nCannot convert values of "{df[field].dtype}" type in df[{field}] to {to_type}.')
            exit()
        except ValueError as e:
            print(
                f'TypeError: {e}\nCannot convert values of "{df[field].dtype}" type in df[{field}] to {to_type}.')
            exit()
    return df


def nans_to_mean(df: pd.DataFrame, *args: str) -> pd.DataFrame:
    """Dataframe and features as input, returns new DF with NANs converted to mean of existing features."""
    for field in args:
        try:
            df[field] = df[field].fillna(df[field].mean())
        except TypeError as e:
            print(
                f'{e}\nTypeError: Could not calculate mean of "{df[field].dtype}" type in "{field}". Try converting to a number type first.')
            sys.exit()
    return df


def dollars_str_to_type_NANs_to_mean(df: pd.DataFrame, *args: str, to_type: str = 'int64') -> pd.DataFrame:
    '''Removes '$' and ',' from dollar values to convert to default int64 or specified type. Returns modified DF.'''
    for dollar_field in args:
        for index, entry in enumerate(df[dollar_field]):
            if type(entry) == str:
                dollar_val = int(entry[1:].replace(',', ''))
                df.at[index, dollar_field] = dollar_val
        df = nans_to_mean(df, dollar_field)
        df = features_to_dtype(df, dollar_field, to_type=to_type)
    return df


def make_ordinal(df: pd.DataFrame, *args: str) -> pd.DataFrame:
    enc = OrdinalEncoder(dtype='uint8')
    for arg in args:
        arr_2d = [[x] for x in df[arg]]
        df[arg] = enc.fit_transform(arr_2d)
    return df


def cat_to_1hot(df: pd.DataFrame, *args: str) -> pd.DataFrame:
    for arg in args:
        df = pd.get_dummies(df, columns=[arg])
    return df


def drop_features(df: pd.DataFrame, *args: str) -> pd.DataFrame:
    for arg in args:
        df = df.drop(arg, axis=1)
    return df


"""
NOTE: There are 600+ nans in 'OCCUPATION' but all have an income, so not unemployed,
could be retired but average age doesnt suggest this,
the majority are educated to masters or phd, data shows BlueCollars dont tend to have this level of education
so not replacing with the mode (blue collar). Decided to create 'Unknown' category.
"""

df['OCCUPATION'] = df['OCCUPATION'].fillna('Unknown')

df = dollars_str_to_type_NANs_to_mean(
    df, 'INCOME', 'HOME_VAL', 'BLUEBOOK', 'OLDCLAIM', 'CLM_AMT')

df = nans_to_mean(df, 'AGE', 'CAR_AGE', 'YOJ')

df = features_to_dtype(df, 'AGE', 'CAR_AGE', 'YOJ')

df = cat_to_1hot(df, 'EDUCATION', 'OCCUPATION', 'CAR_USE', 'CAR_TYPE')

df = make_ordinal(df, 'GENDER', 'PARENT1', 'MSTATUS', 'REVOKED', 'URBANICITY')

# CLM_AMT is > 0 only if CLAIM_FLAG = 1. BIRTH reduntant with AGE. RED_CAR and ID dont correlate with CLAIM_FLAG:
df = drop_features(df, 'CLM_AMT', 'BIRTH', 'RED_CAR', 'ID')


# Engineer some new features:

# df['BLUE%INCOME*'] = df['INCOME'] / df['BLUEBOOK']  # -0.026 - POOR
df['PTS_PER_AGE*'] = df['MVR_PTS'] / df['AGE']      # +0.232 - GREAT!
df['CLM_PER_AGE*'] = df['CLM_FREQ'] / df['AGE']      # +0.234 - GREAT!
df['RICH_KIDS*'] = df['INCOME'] / df['AGE']         # -0.112 - FAIR
# df['HOME_PER_BLUE*'] = df['HOME_VAL'] / df['BLUEBOOK'] # - 0.04 - POOR
df['CLM_PER_MILE*'] = df['CLM_FREQ'] / df['TRAVTIME'] # 0.120 - FAIR

df = features_to_dtype(df, 'PTS_PER_AGE*', 'CLM_PER_AGE*', 'RICH_KIDS*', 'CLM_PER_MILE*')


if __name__ == '__main__':
    df.info()
    pass
