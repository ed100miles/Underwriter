import pickle
import pandas as pd
from datetime import datetime
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import cross_val_score, HalvingGridSearchCV
from typing import List

from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neural_network import MLPClassifier


# Classifiers and parameters: 

svc = (SVC(), {
    'C': [20.0, 22.5, 25.0, 27.5],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'degree': [2, 3, 4],  # polly kernel only param
    'gamma': ['scale', 'auto'],  # rbf only param
    'coef0': [0.0, 0.5, 1.0]
})

nusvc = (NuSVC(), {
    'nu': [0.25, 0.5, 0.75, 1],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'degree': [2, 3, 4],  # polly kernel only param
    'gamma': ['scale', 'auto'],  # rbf only param
    'coef0': [0.0, 0.5, 1.0]
})

lsvc = (LinearSVC(), {
    'C': [0.5, 1, 1.5, 2],
    'loss': ['hinge', 'squared_hinge'],
    'dual': [True, False]
})

sgd = (SGDClassifier(), {
    'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'alpha': [0.0001, 0.0002, 0.0003],
    'l1_ratio': [0.1, 0.15, 0.2],
    'learning_rate': ['optimal', 'invscaling'],
    'eta0': [0.01, 0.1, 1],
    'max_iter': [1000000]
})

knc = (KNeighborsClassifier(), {
    'n_neighbors': [3, 5, 8],
    'weights': ['uniform', 'distance'],
    'leaf_size': [25, 30, 35],
    'p': [1, 2, 3],
})

mlp = (MLPClassifier(), {
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'alpha': [0.00005, 0.0001, 0.00015],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'power_t': [0.25, 0.5, 0.75],
    'max_iter': [10000]
})

gnb = (GaussianNB(), {
    'var_smoothing': [0.0000000001, 0.00000000005, 0.00000000015]
})

bnb = (BernoulliNB(), {
    'alpha': [0, 0.5, 1, 1.5, 2, 2.5],
})

mnb = (MultinomialNB(), {
    'alpha': [0, 0.5, 1, 1.5, 2, 2.5],
})

logr = (LogisticRegression(), {
    'C': [1.25, 1.5, 2],
    'intercept_scaling': [0.5, 0.75, 1],
    # 'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
    'max_iter': [100000]
})

d_tree = (DecisionTreeClassifier(), {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_features': [None, 'auto', 'sqrt', 'log2']
})

rfc = (RandomForestClassifier(), {
    'criterion': ['gini', 'entropy'],
    'max_features': ['auto', 'sqrt', 'log2'],
    'class_weight': [None, 'balanced']
})

ada = (AdaBoostClassifier(), {
    'n_estimators': [75, 80, 85, 90, 95, 100, 105, 110, 115, 120],
    'learning_rate': [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
})

# Testing and logging functions: 

def log_it(filename: str, log: str) -> None:
    with open(filename, 'a') as log_file:
        log_file.write(log)


def log_csv(filename: str, log: str) -> None:
    with open(filename, 'a') as log_csv:
        log_csv.write(log)


def OOS_test(clfs: list, data_X, data_y, tested_params: str) -> tuple:
    for clf in clfs:
        clf.fit(data_X, data_y)
        correct = claims = non_claims = non_claims_predicted = claims_predicted = 0
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

        total_acc = round(correct / len(list(test_y)) * 100, 2)
        claims_acc = round(claims_predicted / claims * 100, 2)
        non_claims_acc = round(non_claims_predicted / non_claims * 100, 2)

        log_it('./data/log.txt',
               f'\n{repr(clf)}\nTested Parameters: {tested_params}\n'
               f'OOS total accuracy: {total_acc}%\n'
               f'Claims prediction accuracy: {claims_acc}%\n'
               f'Non-claims prediction accuracy: {non_claims_acc}%\n')

    return (total_acc, claims_acc, non_claims_acc)


def cross_validate(clfs: list, data_X, data_y):
    for clf in clfs:
        scores = cross_val_score(clf, data_X, data_y)
        log_it(f'{clf} scored:\n{scores}\n')


def test_classifiers(clfs: list, input_data_sets: List[tuple],
                     input_data_names: List[str]) -> None:
    log_it(
        './data/log.txt',
        f'{"*" * 10}     Started New Classifiers Test @'
        f' {pd.to_datetime(datetime.now()).round("1s")}     {"*" * 10}\n\n')

    data = zip(input_data_sets, input_data_names)

    for data_Xy, data_name in data:
        X, y = data_Xy
        
        for classifier in clfs:
            clf, param_grid = classifier

            log_it(
                './data/log.txt', f'\n{"*" * 10}     {clf}     {"*" * 10}'
                f'      {data_name}     {"*" * 10}\n')
            try:
                search = HalvingGridSearchCV(
                    clf, param_grid, n_jobs=3).fit(X, y)
                total_acc, claims_acc, non_claims_acc = OOS_test(
                    [search.best_estimator_], X, y, param_grid)
                log_csv('./data/test_classifier.csv',
                        f'{clf},{data_name},{total_acc},'
                        f'{claims_acc},{non_claims_acc},'
                        # .replace() for csv formatting
                        f'{str(search.best_estimator_).replace(",", "|")},'
                        f'{str(param_grid).replace(",", "|")}\n')
            except Exception as e:
                log_it('./data/error_log.txt', f'{classifier}:\n{e}')


if __name__ == '__main__':

    with open('./data/pickles/data_sets', 'rb') as data_sets_pickle:
        data_sets = pickle.load(data_sets_pickle)

    train_data, test_data, smote_train_data, os_train_data, us_train_data = data_sets

    smote_train_X, smote_train_y = smote_train_data
    os_train_X, os_train_y = os_train_data
    us_train_X, us_train_y = us_train_data
    train_X, train_y = train_data
    test_X, test_y = test_data

    clfs = [bnb, mnb, svc, nusvc, lsvc, sgd,
            knc, mlp, gnb, logr, d_tree, rfc, ada]

    test_classifiers(
        clfs,
        [
            train_data,
            smote_train_data,
            os_train_data,
            us_train_data
        ], [
            'Unbalanced_train_data',
            'Smote_balanced_train_data',
            'Random_over_sampled',
            'Random_under_sampled',
        ])


    # AdaBoost selected, pickle trained model:

    # ada = AdaBoostClassifier(learning_rate=0.4, n_estimators=100)
    # print(OOS_test([ada], us_train_X, us_train_y, 'LR:0.4 | n_est=100'))

    # with open('./data/pickles/models/ada76.pickle', 'wb') as file:
    #     pickle.dump(ada, file)


    pass
