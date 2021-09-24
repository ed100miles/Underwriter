from numpy import dtype
import pandas as pd
from sklearn.utils.validation import check_array
from feature_engineer import df
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle

from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.under_sampling import RandomUnderSampler

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

def log_it(str):
    with open('log.txt', 'a') as log:
        log.write(str)

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

def get_cat_indicies(df):
    cat_indicies = []
    for index, feature in enumerate(df.columns):
        if df[feature].dtype == 'uint8':
            cat_indicies.append(index)
    return cat_indicies
cat_indicies = get_cat_indicies(train_X)


rus = RandomUnderSampler()
ros = RandomOverSampler()
smotenc = SMOTENC(categorical_features=cat_indicies)

smote_train_X, smote_train_y = smotenc.fit_resample(train_X, train_y)
os_train_X, os_train_y = ros.fit_resample(train_X, train_y)
us_train_X, us_train_y = ros.fit_resample(train_X, train_y)

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
gnb = GaussianNB()
gap = GaussianProcessClassifier()
tree = DecisionTreeClassifier()
rfc = RandomForestClassifier()
ada = AdaBoostClassifier()

clfs = [svc, sgd, gnb, knc, mlp, rfc, ada] 
#  mlp, rfc, ada

def OOS_test(clfs:list, data_X, data_y):
    for clf in clfs:
        clf.fit(data_X, data_y)
        correct = 0
        claims = 0
        non_claims = 0
        non_claims_predicted = 0
        claims_predicted = 0
        for x in range(len(list(test_y))):
            prediction = clf.predict([test_X.to_numpy()[x]])
            label = list(test_y)[x]
            if label == 1:
                claims += 1
            else:
                non_claims += 1
            if int(prediction) == label:
                correct += 1
            if int(prediction) == 0 and label == 0:
                non_claims_predicted += 1
            elif int(prediction) == 1 and label == 1:
                claims_predicted += 1
        log_it(f'\n{repr(clf)}')
        log_it(f'\nOOS total accuracy: {round(correct/len(list(test_y))*100, 2)}%\n\
Claims prediction accuracy: {round(claims_predicted / claims * 100, 2)}%\n\
Non-claims prediction accuracy: {round(non_claims_predicted / non_claims * 100, 2)}%\n')

def cross_validate(clfs:list, data_X, data_y):
    for clf in clfs:
        scores = cross_val_score(clf, data_X, data_y)
        log_it(f'{clf} scored:')
        log_it(scores, '\n')


if __name__ == '__main__':
    # print(train_X)
    # # cross_validate(clfs, os_train_X, os_train_y)
    # # cross_validate(clfs, train_X, train_y)

    log_it(f'\n\n{"*"*20} "Unbalanced Data" {"*"*20}\n')
    OOS_test(clfs, train_X, train_y)

    log_it(f'\n\n{"*"*20} "Random Over Sampled Data" {"*"*20}\n')
    OOS_test(clfs, os_train_X, os_train_y)

    log_it(f'\n\n{"*"*20} "Random Under Sampled Data" {"*"*20}\n')
    OOS_test(clfs, us_train_X, us_train_y)

    log_it(f'\n\n{"*"*20} "SMOTENC Sampled Data" {"*"*20}\n')
    OOS_test(clfs, smote_train_X, smote_train_y)
    
    pass

# 5, 7, 8 