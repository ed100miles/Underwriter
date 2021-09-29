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

    def format_str(self, df):
        for dollar_field in self.dollar_features:
            '''Remove $'s and commas from string '''
            for index, entry in enumerate(df[dollar_field]):
                if type(entry) == str:
                    dollar_val = int(entry.replace(',', '').replace('$', ''))
                    df.at[index, dollar_field] = dollar_val
        return df

    def transform(self, df: pd.DataFrame):
        df = self.format_str(df)
        for dollar_field in self.dollar_features:
            '''Convert to specified dtype'''
            try:
                df[dollar_field] = df[dollar_field].astype(
                    self.features_to_dtype)
            except TypeError as e:
                print(
                    f'TypeError: {e}\nCannot convert values of'
                    f' "{df[dollar_field].dtype}" type in df[{dollar_field}]'
                    f' to {self.features_to_dtype}.')
                sys.exit()
            except ValueError as e:
                print(
                    f'TypeError: {e}\nCannot convert values of'
                    f' "{df[dollar_field].dtype}" type in df[{dollar_field}]'
                    f' to {self.features_to_dtype}.')
                sys.exit()

            '''Convert nans in field mean'''
            if self.nans_to_mean:
                try:
                    df[dollar_field] = df[dollar_field].fillna(
                        df[dollar_field].mean())
                except TypeError as e:
                    print(
                        f'{e}\nTypeError: Could not calculate mean of'
                        f' "{df[dollar_field].dtype}" type in'
                        f' "{dollar_field}". Try converting to a number'
                        f' type first.')
                    sys.exit()

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
                    f'{e}\nTypeError: Could not calculate mean of '
                    f'"{df[feature].dtype}" type in "{feature}".' 
                    f'Try converting to a number type first.')
                sys.exit()
        return df


class Scale(BaseEstimator, TransformerMixin):
    def __init__(self, feats_to_scale: List[str]):
        self.feats_to_scale = feats_to_scale
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature in self.feats_to_scale:
            df[feature] = self.scaler.fit_transform([[x] for x in df[feature]])
        return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(
        self, 
        pts_per_age:bool=True, 
        clm_per_age:bool=True, 
        clm_per_mile:bool=False
        ):

        self.pts_per_age = pts_per_age
        self.clm_per_age = clm_per_age
        self.clm_per_mile = clm_per_mile

    def fit(self, X, y=None):
        return self
    
    def transform(self, df:pd.DataFrame) -> pd.DataFrame:
        if self.pts_per_age:
            df['PTS_PER_AGE*'] = df['MVR_PTS'] / df['AGE']
        if self.clm_per_age:
            df['CLM_PER_AGE*'] = df['CLM_FREQ'] / df['AGE'] 
        if self.clm_per_mile:
            df['CLM_PER_MILE*'] = df['CLM_FREQ'] / df['TRAVTIME']
        return df


if __name__ == '__main__':

    df = pd.read_csv('./data/car_insurance_claim.csv')

    train_set, test_set = train_test_split(df, test_size=0.1, random_state=42)
    train_set = train_set.reset_index()
    test_set = test_set.reset_index()

    # Define pipeline parameters:

    dollar_fields = ['INCOME', 'HOME_VAL', 'BLUEBOOK', 'OLDCLAIM', 'CLM_AMT']
    drop_features = ['CLM_AMT', 'BIRTH', 'RED_CAR', 'ID', 'index']
    features_to_1hot = ['EDUCATION', 'OCCUPATION', 'CAR_USE', 'CAR_TYPE']
    nans_to_mean = ['AGE', 'CAR_AGE', 'YOJ']
    features_to_ordinal = ['GENDER', 'PARENT1',
                           'MSTATUS', 'REVOKED', 'URBANICITY']

    non_binary_features = [column for column in train_set.columns if len(
        train_set[column].unique()) > 2]  # gets non-binary features

    features_to_scale = list(set(non_binary_features) -
                             set(drop_features + features_to_1hot))

    data_pipeline = Pipeline([
        ('dollar_transform', DollarTranformer(dollar_fields)),
        ('drop_features', DropFeatures(drop_features)),
        ('feat_to_1hot', Make1Hot(features_to_1hot)),
        ('feat_to_ordinal', MakeOrdinal(features_to_ordinal)),
        ('feat_to_mean', NaN2Mean(nans_to_mean)),
        ('build_feats', FeatureEngineer()),
        ('feat_scale', Scale(features_to_scale))
    ])

    train_set = data_pipeline.fit_transform(train_set)
    test_set = data_pipeline.transform(test_set)

    train_y, test_y = train_set['CLAIM_FLAG'], test_set['CLAIM_FLAG']
    train_X = train_set.drop(['CLAIM_FLAG'], axis=1)
    test_X = test_set.drop(['CLAIM_FLAG'], axis=1)
    train_data = (train_X, train_y)
    test_data = (test_X, test_y)

    # Balance data sets:

    cat_features = [index for index, feature 
                    in enumerate(train_X.columns) 
                    if len(train_X[feature].unique()) == 2]

    rus = RandomUnderSampler()
    ros = RandomOverSampler()
    smotenc = SMOTENC(categorical_features=cat_features)
    
    smote_train_data = smotenc.fit_resample(train_X, train_y)
    os_train_data = ros.fit_resample(train_X, train_y)
    us_train_data = ros.fit_resample(train_X, train_y)

    data_sets = [train_data, test_data, smote_train_data, os_train_data, us_train_data]

    with open('./data/pickles/data_sets', 'wb') as pickle_out:
        pickle.dump(data_sets, pickle_out)

