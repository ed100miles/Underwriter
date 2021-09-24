from prepare_data import test_X, test_y, train_X, train_y, smote_train_X, smote_train_y, os_train_X, os_train_y, us_train_X, us_train_y
from sklearn.model_selection import cross_val_score

# model imports:
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def log_it(filename:str, log:str) -> None:
    with open(filename, 'a') as log_file:
        log_file.write(log)


svc = svm.SVC()
sgd = SGDClassifier()
knc = KNeighborsClassifier()
mlp = MLPClassifier()
gnb = GaussianNB()
gap = GaussianProcessClassifier()
tree = DecisionTreeClassifier()
rfc = RandomForestClassifier()
ada = AdaBoostClassifier()

clfs = [svc]
#  mlp, rfc, ada


def OOS_test(clfs: list, data_X, data_y):
    for clf in clfs:
        clf.fit(data_X, data_y)
        correct = claims = non_claims = non_claims_predicted = claims_predicted = 0
        for x in range(len(list(test_y))):
            prediction = clf.predict([test_X.to_numpy()[x]])
            label = list(test_y)[x]
            if label == 1: claims += 1
            else: non_claims += 1
            if int(prediction) == label: correct += 1
            if int(prediction) == 0 and label == 0: non_claims_predicted += 1
            elif int(prediction) == 1 and label == 1: claims_predicted += 1
        log_it('log.txt', f'\n{repr(clf)}\n\
OOS total accuracy: {round(correct/len(list(test_y))*100, 2)}%\n\
Claims prediction accuracy: {round(claims_predicted / claims * 100, 2)}%\n\
Non-claims prediction accuracy: {round(non_claims_predicted / non_claims * 100, 2)}%\n')


def cross_validate(clfs: list, data_X, data_y):
    for clf in clfs:
        scores = cross_val_score(clf, data_X, data_y)
        log_it(f'{clf} scored:\n{scores}\n')


if __name__ == '__main__':

    # # cross_validate(clfs, os_train_X, os_train_y)
    # # cross_validate(clfs, train_X, train_y)

    log_it('log.txt', f'\n\n{"*"*20} "Unbalanced Data" {"*"*20}\n')
    OOS_test(clfs, train_X, train_y)

    log_it('log.txt', f'\n\n{"*"*20} "Random Over Sampled Data" {"*"*20}\n')
    OOS_test(clfs, os_train_X, os_train_y)

    log_it('log.txt', f'\n\n{"*"*20} "Random Under Sampled Data" {"*"*20}\n')
    OOS_test(clfs, us_train_X, us_train_y)

    log_it('log.txt', f'\n\n{"*"*20} "SMOTENC Sampled Data" {"*"*20}\n')
    OOS_test(clfs, smote_train_X, smote_train_y)

