from imblearn import over_sampling
import pandas as pd
from pandas.core.frame import DataFrame
from feature_engineer import df
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle

from imblearn.over_sampling import RandomOverSampler

# model imports:
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


scaler = MinMaxScaler()

def scale_data(df, scale:list):
    for to_scale in scale:
        df[to_scale] = scaler.fit_transform([[x] for x in df[to_scale]])
    return df

to_scale_list = [x for x in df.columns if df[x].dtype == 'float64' 
                or df[x].dtype == 'int64']

df = scale_data(df, to_scale_list)

# print(df['INCOME'].head())
# print(df['AGE'].head())

train_set, test_set = train_test_split(df, test_size=0.1, random_state=42)

# consider stratified samples using StraifiedShuffleSplit?

train_y, test_y = train_set['CLAIM_FLAG'], test_set['CLAIM_FLAG']
train_X = train_set.drop(['CLAIM_FLAG'], axis=1)
test_X = test_set.drop(['CLAIM_FLAG'], axis=1)

ros = RandomOverSampler()
os_train_X, os_train_y = ros.fit_resample(train_X, train_y)

# print(len(train_y))
# print(train_y.head())

# claim_count = 0
# non_claim_count = 0
# for entry in os_train_y:
#     if entry == 1:
#         claim_count += 1 
#     else: 
#         non_claim_count += 1
# print(claim_count)
# print(non_claim_count)
# print(f'balance:{claim_count/non_claim_count*100}%')

# train_data = pd.concat([train_X, train_y], axis=1)
# train_data = shuffle(train_data)

# grouped = train_data.groupby('CLAIM_FLAG')
# print(grouped)


# def count_claim_vs_non_claim(df):
#     claims = 0
#     non_claims = 0
#     for index, row in df.iterrows():
#         if row['CLAIM_FLAG'] == 1:
#             claims += 1
#         else:
#             non_claims += 1
#     print('claims:',claims, ' non-claims:', non_claims)

# count_claim_vs_non_claim(train_data)

# def balance_data(df):
#     out_df = pd.DataFrame()
#     claims = 0
#     non_claims = 0
#     for _, row in df.iterrows():
#         if row['CLAIM_FLAG'] == 1:
#             if claims <= non_claims:
#                 out_df = pd.concat([out_df, row])
#                 claims += 1
#         else:
#             if non_claims <= claims:
#                 out_df = pd.concat([out_df, row])
#                 non_claims += 1
#     return out_df
# balanced_df = balance_data(train_data)
# print(balanced_df.head())

# TODO: Balance the training data, think it's throwing it off !!!!

# print(list(test_y))

svc = svm.SVC()
sgd = SGDClassifier()
knc = KNeighborsClassifier()
mlp = MLPClassifier()
gap = GaussianProcessClassifier()
tree = DecisionTreeClassifier()
rfc = RandomForestClassifier()
ada = AdaBoostClassifier()

clfs = [svc, sgd, knc, mlp, gap, tree, rfc, ada]

def OOS_test(clfs:list, data_X, data_y):
    for clf in clfs:
        clf.fit(data_X, data_y)
        correct = 0
        non_claims_predicted = 0
        claims_predicted = 0
        for x in range(1000):
            prediction = clf.predict([test_X.to_numpy()[x]])
            # print(f'predicted:{prediction}')
            label = list(test_y)[x]
            # print(f'label:{label}')
            if int(prediction) == int(label):
                correct += 1
            if int(prediction) == 0:
                non_claims_predicted += 1
            else:
                claims_predicted += 1
        print(clf)
        print(f'\nOOS accuracy: {correct/1000*100}%\n\
    claims predicted:{claims_predicted}\n\
    non-claims predicted:{non_claims_predicted}\n')
        print(f'claim/non-claim prediction ratio: {non_claims_predicted/claims_predicted}\n')

def cross_validate(clfs:list, data_X, data_y):
    for clf in clfs:
        scores = cross_val_score(clf, data_X, data_y)
        print(f'{clf} scored:')
        print(scores, '\n')

if __name__ == '__main__':

    cross_validate(clfs, train_X, train_y)
    OOS_test(clfs, train_X, train_y)

    print('*********************')

    cross_validate(clfs, os_train_X, os_train_y)
    OOS_test(clfs, os_train_X, os_train_y)

