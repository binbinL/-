import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB #伯努利型
from sklearn import svm


def deal():
    data = pd.read_csv('D:/pythoncode_2/Bigdata_exp/BD/loan_predict/train_dataset/train_public.csv')
    test = pd.read_csv('D:/pythoncode_2/Bigdata_exp/BD/loan_predict/test_public.csv')

    ID = test['loan_id']

    # pd.set_option('display.max_columns', None)    # 显示所有列
    # print(data.head(10))

    # dupNum = data.shape[0] - data.drop_duplicates().shape[0]
    # print("数据集中有%s列重复值" % dupNum)
    #
    # dupNum = test.shape[0] - test.drop_duplicates().shape[0]
    # print("数据集中有%s列重复值" % dupNum)

    work_year_list = {
        '< 1 year': 0,
        '1 year': 1,
        '2 years': 2,
        '3 years': 3,
        '4 years': 4,
        '5 years': 5,
        '6 years': 6,
        '7 years': 7,
        '8 years': 8,
        '9 years': 9,
        '10+ years': 10,
    }

    data['work_year'] = data['work_year'].map(work_year_list)
    test['work_year'] = test['work_year'].map(work_year_list)

    class_list = {
        'A': 1,
        'B': 2,
        'C': 3,
        'D': 4,
        'E': 5,
        'F': 6,
        'G': 7,
    }
    data['class'] = data['class'].map(class_list)
    test['class'] = test['class'].map(class_list)

    # 无值情况
    data['work_year'] = data['work_year'].fillna(-1)
    test['work_year'] = test['work_year'].fillna(-1)

    data['pub_dero_bankrup'] = data['pub_dero_bankrup'].fillna(-1)
    test['pub_dero_bankrup'] = test['pub_dero_bankrup'].fillna(-1)

    # data['class'] = data['class'].fillna(-1)
    # test['class'] = test['class'].fillna(-1)

    # 填均值
    for i in ['f0', 'f1', 'f2', 'f3', 'f4']:
        data[i] = data[i].fillna(data[i].mean())
        test[i] = test[i].fillna(test[i].mean())

    # data.dropna(axis=0, how='any', inplace=True)

    for i in ['loan_id', 'user_id', 'issue_date', 'earlies_credit_mon', 'title', 'post_code', 'region']:
        data = data.drop(i, axis=1)
        test = test.drop(i, axis=1)

    T_cols = ['employer_type', 'industry']
    for col in T_cols:
        lbl = LabelEncoder().fit(data[col])
        data[col] = lbl.transform(data[col])
        test[col] = lbl.transform(test[col])

    # print("缺失test：")
    # print(test.isnull().any())
    # print("缺失train：")
    # print(data.isnull().any())

    # 数据标准化
    std_col = ['total_loan', 'interest', 'monthly_payment', 'scoring_low', 'scoring_high', 'recircle_b', 'recircle_u',
               'debt_loan_ratio', 'f0', 'f1', 'f2', 'f3', 'f4', 'early_return_amount', 'early_return_amount_3mon']
    for i in std_col:
        data[i] = (data[i] - min(data[i])) / (max(data[i]) - min(data[i]))*100
        test[i] = (test[i] - min(test[i])) / (max(test[i]) - min(test[i]))*100

    data_label = data['isDefault']
    labels = data_label.values.tolist()
    data = data.drop('isDefault', axis=1)
    data_list = data.values.tolist()
    test_list = test.values.tolist()

    pd.set_option('display.max_columns', None)  # 显示所有列
    print(data.head(20))

    data_list = np.array(data_list)
    labels = np.array(labels)
    test_list = np.array(test_list)

    return data_list, labels, test_list, ID


def SVM(X_train, labels_train, X_test):
    clf = svm.SVC(C=5, kernel='rbf')
    clf.fit(X_train, labels_train)
    pred = clf.predict(X_test)
    return pred


def Logistic(X_train, labels_train, X_test):
    lr = LogisticRegression(max_iter=10000)
    lr.fit(X_train, labels_train)
    pred = lr.predict(X_test)
    return pred


def Naive_Bayes(X_train, labels_train, X_test):
    # wight = [2,2,2,3,1,2,2,2,3,1,2,3,3,2,2,1,2,1,1,1,1,1,1,1,1,1,1,1,2,2,2]
    # wight = np.array(wight)
    # wight = wight/wight.sum()
    # print(wight)

    nb = GaussianNB()
    nb.fit(X_train, labels_train)
    pred = nb.predict(X_test)
    return pred

def BNB(X_train, labels_train, X_test):
    nb = BernoulliNB()
    nb.fit(X_train, labels_train)
    pred = nb.predict(X_test)
    return pred


def save(ID, res):
    a = []
    for i in ID:
        a.append(i)
    dataframe = pd.DataFrame({'id': a, 'isDefault': res})
    dataframe.to_csv("submission.csv", index=False, sep=',')


if __name__ == '__main__':
    X_train, labels_train, X_test, id = deal()
    print(X_train.shape)
    print(labels_train.shape)
    print(X_test.shape)

    #res = Logistic(X_train, labels_train, X_test)  # 0.57
    #res = Naive_Bayes(X_train, labels_train, X_test) #0.805
    # res = SVM(X_train, labels_train, X_test)

    res = BNB(X_train, labels_train, X_test)
    save(id, res)
