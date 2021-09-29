
from typing import List
import pandas as pd
import sys
import pickle

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.under_sampling import RandomUnderSampler

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer



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


def scale_data(df: pd.DataFrame, scale: list) -> pd.DataFrame:
    scaler = MinMaxScaler()
    for to_scale in scale:
        df[to_scale] = scaler.fit_transform([[x] for x in df[to_scale]])
    return df


# transformer = FunctionTransformer(np.log1p, validate=True)

# dollars_format = FunctionTransformer(
#     dollars_str_to_type_NANs_to_mean,
#     dollar_fields=[]
# )

if __name__ == '__main__':

    df = pd.read_csv('./data/car_insurance_claim.csv')

    df1 = pd.DataFrame(df.loc[:9000,:])
    df2 = pd.DataFrame(df.loc[9000:,:])


    dollar_fields = ['INCOME', 'HOME_VAL', 'BLUEBOOK', 'OLDCLAIM', 'CLM_AMT']
    drop_features = ['CLM_AMT', 'BIRTH', 'RED_CAR', 'ID']
    features_to_1hot = ['EDUCATION', 'OCCUPATION', 'CAR_USE', 'CAR_TYPE']
    nans_to_mean = ['AGE', 'CAR_AGE', 'YOJ'] + dollar_fields
    features_to_ordinal = ['GENDER', 'PARENT1',
                           'MSTATUS', 'REVOKED', 'URBANICITY']
    non_binary_features = [column for column in df1.columns if len(
        df1[column].unique()) > 2]  # gets non-binary features

    features_to_scale = list(set(non_binary_features) -
                             set(drop_features + features_to_1hot))

    dollars = FunctionTransformer(
            dollars_str_to_type_NANs_to_mean
        )

    data_pipe = Pipeline([
        ('dollar_transform', dollars)
    ])

    # data_pipeline = Pipeline([
    #     ('dollar_transform', DollarTranformer(dollar_fields)),
    #     ('feat_to_mean', NaN2Mean(nans_to_mean)),
    #     ('drop_features', DropFeatures(drop_features)),
    #     ('feat_to_1hot', Make1Hot(features_to_1hot)),
    #     ('feat_to_ordinal', MakeOrdinal(features_to_ordinal)),
    #     ('feat_scale', Scale(features_to_scale))
    # ])

    trans_df = data_pipe.fit_transform(df1)
    # trans_df = data_pipeline.transform(df2)

    trans_df.info()
