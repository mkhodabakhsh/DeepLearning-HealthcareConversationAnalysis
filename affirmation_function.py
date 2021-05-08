import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def affirmation_train_and_test(path_test):

    balance_ratio = 3

    current_dir = os.getcwd()
    data = pd.read_csv(current_dir + '/dataframe_train_affirm.csv') 
    y = data["affirm"].tolist()
    X_train_raw = data.drop(['sentence', 'open_q', 'close_q', 'reflect'], axis=1)#, 'ny', 'qw', 'nn', '^2', 'qo', 'qh', '^h', 'ar', 'ng', 'br', 'no', 'qrr', 't3', 'oo_co_cc', 'aap_am', 't1', 'bd', '^g', 'qw^d', 'fa', 'ft'], axis=1)

    #X_train_raw, X_test, y_trash, y_test = train_test_split(X, y,stratify=y, test_size = 0.001)
    
    dict_={}
    data_tr =X_train_raw.reset_index(drop=True)
    aug_data = pd.DataFrame()
    index_ = data_tr.index

    aug_data = data_tr[data_tr['affirm']==1]
    data_tr_bal = data_tr

    for i in range(0,len(data_tr.index)//len(aug_data.index)//balance_ratio):
        data_tr_bal=pd.concat([aug_data,data_tr_bal])
    data_tr_bal =data_tr_bal.reset_index(drop=True)

 
    ####################################################
    y_ = data_tr_bal["affirm"].tolist()
    X_ = data_tr_bal.drop(['affirm'], axis=1)

    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(X_, y_)
    ####################################################

    y_pred_tr = svclassifier.predict(X_)

    #print(confusion_matrix(y_,y_pred_tr))
    #print(classification_report(y_,y_pred_tr))

    ####################################################

    data_test = pd.read_csv(path_test) 
    data_test=data_test.drop(['Unnamed: 0'],axis=1)

    y_test = data_test["affirm"].tolist()
    X_test = data_test.drop(['sentence', 'open_q', 'close_q', 'reflect'], axis=1)

    X_test  = X_test.reset_index(drop=True)
    X_test_ = X_test.drop(['affirm'], axis=1)
    y_pred_ = list(svclassifier.predict(X_test_))
        
    print(confusion_matrix(y_test,y_pred_))
    #print(classification_report(y_test,y_pred_))

    return confusion_matrix(y_test,y_pred_) , y_pred_