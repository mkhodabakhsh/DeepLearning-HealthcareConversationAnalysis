
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier


def reflection_train_and_test(owd , path_test, test_comments):

    balance_ratio = 3
    os.chdir(owd)
    cwd = os.getcwd()

    data = pd.read_csv(cwd + '/data/dataframe_train_reflect.csv' , encoding = "ISO-8859-1") 

    y = data["reflect"].tolist()
    X_train_raw = data.drop(['sentence', 'open_q', 'close_q', 'affirm'], axis=1)
    dict_={}
    data_tr =X_train_raw.reset_index(drop=True)
    aug_data = pd.DataFrame()
    index_ = data_tr.index

    aug_data = data_tr[data_tr['reflect']==1]
    data_tr_bal = data_tr

    for i in range(0,len(data_tr.index)//len(aug_data.index)//balance_ratio):
        data_tr_bal=pd.concat([aug_data,data_tr_bal])

    data_tr_bal =data_tr_bal.reset_index(drop=True)

    ####################################################

    y_ = data_tr_bal["reflect"].tolist()
    X_ = data_tr_bal.drop(['reflect'], axis=1)
    try:X_=X_.drop(['Unnamed: 0'],axis=1)
    except: pass

    svclassifier = SVC(kernel='rbf',gamma='auto')
    svclassifier.fit(X_, y_)


    adclassifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                            n_estimators=200)
    adclassifier = adclassifier.fit(X_, y_)


    gbclassifier = GradientBoostingClassifier()
    gbclassifier = gbclassifier.fit(X_, y_)

    dtclassifier = tree.DecisionTreeClassifier()
    dtclassifier = dtclassifier.fit(X_, y_)

    rfclassifier = RandomForestClassifier(n_estimators=500)
    rfclassifier = rfclassifier.fit(X_, y_)

    etclassifier = ExtraTreesClassifier(n_estimators=500)
    etclassifier = etclassifier.fit(X_, y_)

    mlpclassifier = MLPClassifier(solver='lbfgs', alpha=1e-3,
                        hidden_layer_sizes=(5, 2), random_state=1)
    mlpclassifier.fit(X_, y_)

    ############################################################

    data_test = pd.read_csv(path_test) 
    try:data_test=data_test.drop(['Unnamed: 0'],axis=1)
    except: pass
    
    y_test = data_test["reflect"].tolist()
    X_test = data_test.drop(['sentence', 'open_q', 'close_q', 'affirm'], axis=1)

    X_test  = X_test.reset_index(drop=True)
    X_test_ = X_test.drop(['reflect'], axis=1)
    try:X_test_=X_test_.drop(['Unnamed: 0.1'],axis=1)
    except: pass
    y_pred_dt = list(dtclassifier.predict(X_test_))
    y_pred_sv = list(svclassifier.predict(X_test_))
    y_pred_rf = list(rfclassifier.predict(X_test_))
    y_pred_ad = list(adclassifier.predict(X_test_))
    y_pred_et = list(etclassifier.predict(X_test_))
    y_pred_gb = list(gbclassifier.predict(X_test_))

    y_pred_ = []
    zipped = zip(y_pred_dt,y_pred_sv,y_pred_rf,y_pred_ad,y_pred_et,y_pred_gb)
    for dt,sv,rf,ad,et,gb in zipped:
        y_pred_.append(min((dt+rf+ad+sv+gb) ,1))

    print(confusion_matrix(y_test,y_pred_))

    token_reflect = ['willingness' , 'motivated' , 'motivation' , 'improvements' , 'excellent' , 'Excellent' ,
                 'really' , 'showed' , 'you\'ve' , 'You\'ve' , 'mentioned' , 'told' , 'feel' , 'feeling' ,
                'You\'re' , 'you\'re' , 'willing' , 'handle' , 'able', 'improve' , 'improved' , 'said' ,'saying' , 'think' ,
                 'demonstrated' , 'demonstrate' , 'trusted' , 'trust' , 'looks' , 'hear' , 'hearing' , 
                 'wish' , 'seems' , 'meant' ,  'didn\'t' , 'did' , 'helped' ]
    
    token_existence_ref = []
    for i in range(len(test_comments)):
        token_existence_ref.append(sum([int(token_ in test_comments[i]) for token_ in token_reflect]) )   



    return confusion_matrix(y_test,y_pred_) , y_pred_, token_existence_ref