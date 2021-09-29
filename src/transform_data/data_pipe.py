
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


class DollarTranformer(BaseEstimator, TransformerMixin):
    def __init__(self, dollar_features: List[str], nans_to_mean: bool = True,
                 features_to_dtype: str = 'float64'):

        self.dollar_features = dollar_features
        self.nans_to_mean = nans_to_mean
        self.features_to_dtype = features_to_dtype

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame):
        for dollar_field in self.dollar_features:

            '''Remove $'s and commas from string '''
            for index, entry in enumerate(df[dollar_field]):
                if type(entry) == str:
                    dollar_val = int(entry.replace(',', '').replace('$', ''))
                    df.at[index, dollar_field] = dollar_val

            '''Convert to specified dtype'''
            try:
                df[dollar_field] = df[dollar_field].astype(
                    self.features_to_dtype)
            except TypeError as e:
                print(
                    f'TypeError:  {e}\nCannot convert values of "{df[dollar_field].dtype}" type in df[{dollar_field}] to {self.features_to_dtype}.')
                sys.exit()
            except ValueError as e:
                print(
                    f'ValueError: Dollar() {e}\nCannot convert values of "{df[dollar_field].dtype}" type in df[{dollar_field}] to {self.features_to_dtype}.')
                sys.exit()

            # '''Convert nans in field to mean'''
            # if self.nans_to_mean:
            #     try:
            #         df[dollar_field] = df[dollar_field].fillna(
            #             df[dollar_field].mean())
            #     except TypeError as e:
            #         print(
            #             f'{e}\nTypeError: Could not calculate mean of "{df[dollar_field].dtype}" type in "{dollar_field}". Try converting to a number type first.')
            #         sys.exit()
        return df


class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, drop_features: List[str]):
        self.drop_features = drop_features

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame):
        for feature in self.drop_features:
            df = df.drop(feature, axis=1)
        return df


class Make1Hot(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_1hot: List[str]):
        self.features_to_1hot = features_to_1hot

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame):
        for feature in self.features_to_1hot:
            df = pd.get_dummies(df, columns=[feature])
        return df


class MakeOrdinal(BaseEstimator, TransformerMixin):
    def __init__(self, feats_to_ordinal: List[str]):
        self.feats_to_ordinal = feats_to_ordinal
        self.ordinal_enc = OrdinalEncoder(dtype='uint8')

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature in self.feats_to_ordinal:
            arr_2d = [[x] for x in df[feature]]
            df[feature] = self.ordinal_enc.fit_transform(arr_2d)
        
        return df


class NaN2Mean(BaseEstimator, TransformerMixin):
    def __init__(self, feats_to_mean: List[str]):
        self.feats_to_mean = feats_to_mean

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature in self.feats_to_mean:
            try:
                df[feature] = df[feature].fillna(df[feature].mean())
            except TypeError as e:
                print(
                    f'{e}\nNaN2Mean TypeError: Could not calculate mean of "{df[feature].dtype}" type in "{feature}". Try converting to a number type first.')
                sys.exit()
        return df


class Scale(BaseEstimator, TransformerMixin):
    def __init__(self, feats_to_scale: List[str]):
        self.feats_to_scale = feats_to_scale
        self.scaler = MinMaxScaler()
        self.scalers = {}

    def fit(self, df:pd.DataFrame, y=None):
        self.scalers = {} # auto reset scalers if fit called
        for feature in self.feats_to_scale:
            self.scalers[feature] = MinMaxScaler()
            self.scalers[feature].fit([[x] for x in df[feature]])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature in self.feats_to_scale:
            df[feature] = self.scalers[feature].transform([[x] for x in df[feature]])
        return df

# def get_scalable_features(df, drop_features):


if __name__ == '__main__':

    df = pd.read_csv('./data/car_insurance_claim.csv')

    df1 = pd.DataFrame(df.loc[:10300, :]).reset_index()
    df2 = pd.DataFrame(df.loc[10300:, :]).reset_index()

    dollar_fields = ['INCOME', 'HOME_VAL', 'BLUEBOOK', 'OLDCLAIM', 'CLM_AMT']
    drop_features = ['CLM_AMT', 'BIRTH', 'RED_CAR', 'ID']
    features_to_1hot = ['EDUCATION', 'OCCUPATION', 'CAR_USE', 'CAR_TYPE']
    nans_to_mean = ['AGE', 'CAR_AGE', 'YOJ'] + dollar_fields
    features_to_ordinal = ['GENDER', 'PARENT1',
                           'MSTATUS', 'REVOKED', 'URBANICITY']
    non_binary_features = [column for column in df1.columns if len(
        df1[column].unique()) > 2]  # gets non-binary features

    features_to_scale = list(set(non_binary_features) -
                             set(drop_features + features_to_1hot + ['index']))

    data_pipeline = Pipeline([
        ('dollar_transform', DollarTranformer(dollar_fields)),
        ('feat_to_mean', NaN2Mean(nans_to_mean)),
        ('drop_features', DropFeatures(drop_features)),
        ('feat_to_1hot', Make1Hot(features_to_1hot)),
        ('feat_to_ordinal', MakeOrdinal(features_to_ordinal)),
        ('feat_scale', Scale(features_to_scale))
    ])

    trans_df1 = data_pipeline.fit_transform(df1)
    trans_df2 = data_pipeline.transform(df2)
