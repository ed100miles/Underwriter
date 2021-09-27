import pickle
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import cross_val_score, GridSearchCV, HalvingGridSearchCV
from typing import List
# model imports:
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

with open('./data/pickles/data_sets', 'rb') as data_sets_pickle:
    data_sets = pickle.load(data_sets_pickle)

train_data, test_data, smote_train_data, os_train_data, us_train_data = data_sets

smote_train_X, smote_train_y = smote_train_data
os_train_X, os_train_y = os_train_data
us_train_X, us_train_y = us_train_data
train_X, train_y = train_data
test_X, test_y = test_data

svc = (svm.SVC(), {
    'C': [20.0, 22.5, 25.0, 27.5],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'degree': [2, 3, 4],  # polly kernel only param
    'gamma': ['scale', 'auto'],  # rbf only param
    'coef0': [0.0, 0.5, 1.0]
})

sgd = (SGDClassifier(), {
    'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'alpha': [0.0001, 0.00005, 0.00015],
    'l1_ratio': [0.1, 0.15, 0.2],
    'learning_rate': ['optimal', 'invscaling']
})

knc = (KNeighborsClassifier(),{
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
    'power_t': [0.25, 0.5, 0.75]
})

gnb = (GaussianNB(), {
    'var_smoothing': [0.0000000001, 0.00000000005, 0.00000000015]
})


d_tree = (DecisionTreeClassifier(),{
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_features': [None, 'auto', 'sqrt', 'log2']
})

rfc = (RandomForestClassifier(),{
    'criterion': ['gini', 'entropy'],
    'max_features': ['auto', 'sqrt', 'log2'],
    'class_weight': [None, 'balanced']
})

ada = (AdaBoostClassifier(), {
    'n_estimators': [25, 50, 75],
    'learning_rate': [0.5, 1.0, 1.5]
})

clfs = [svc
#, sgd, knc, mlp, gnb, d_tree, rfc, ada
]
#  mlp, rfc, ada


def log_it(filename: str, log: str) -> None:
    with open(filename, 'a') as log_file:
        log_file.write(log)


def OOS_test(clfs: list, data_X, data_y, tested_params:str):
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
        log_it('log.txt', f'\n{repr(clf)}\n\
Tested Parameters: {tested_params}\n\
OOS total accuracy: {round(correct/len(list(test_y))*100, 2)}%\n\
Claims prediction accuracy: {round(claims_predicted / claims * 100, 2)}%\n\
Non-claims prediction accuracy: {round(non_claims_predicted / non_claims * 100, 2)}%\n')


def cross_validate(clfs: list, data_X, data_y):
    for clf in clfs:
        scores = cross_val_score(clf, data_X, data_y)
        log_it(f'{clf} scored:\n{scores}\n')

def test_classifiers(clfs: list, input_data_sets:List[tuple], input_data_names:List[str]):
    data = zip(input_data_sets, input_data_names)
    for data_Xy, data_name in data:
        X, y = data_Xy
        for classifier in clfs:
            clf, param_grid = classifier
            log_it('log.txt', f'{"*"*10}{" "*5}{clf}{" "*5}{"*"*10}{" "*5}{data_name}{" "*5}{"*"*10}')
            try:
                search = HalvingGridSearchCV(clf, param_grid).fit(X, y)
                print(search.best_estimator_)
                OOS_test([search.best_estimator_], X, y, param_grid)
            except Exception as e:
                log_it('error_log.txt', f'{classifier}:\n{e}')


if __name__ == '__main__':
    test_classifiers(clfs, [train_data, smote_train_data, os_train_data, us_train_data],
    ['train_data', 'smote_train_data', 'os_train_data', 'us_train_data'])




    # for classifier in clfs:
    #     for data_type in data_sets:
    #         try:
    #             clf, param_grid = classifier
    #             search = HalvingGridSearchCV(clf, param_grid).fit(
    #                 smote_train_X, smote_train_y)
    #             print(search.best_estimator_)
    #             OOS_test([search.best_estimator_], smote_train_X, smote_train_y, param_grid)
    #         except Exception as e:
    #             log_it('error_log.txt', f'{classifier}:\n{e}')



    # # cross_validate(clfs, os_train_X, os_train_y)
    # # cross_validate(clfs, train_X, train_y)

    # log_it('log.txt', f'\n\n{"*"*20} "Unbalanced Data" {"*"*20}\n')
    # OOS_test(clfs, train_X, train_y)

    # log_it('log.txt', f'\n\n{"*"*20} "Random Over Sampled Data" {"*"*20}\n')
    # OOS_test(clfs, os_train_X, os_train_y)

    # log_it('log.txt', f'\n\n{"*"*20} "Random Under Sampled Data" {"*"*20}\n')
    # OOS_test(clfs, us_train_X, us_train_y)

    # log_it('log.txt', f'\n\n{"*"*20} "SMOTENC Sampled Data" {"*"*20}\n')
    # OOS_test(clfs, smote_train_X, smote_train_y)
    pass

