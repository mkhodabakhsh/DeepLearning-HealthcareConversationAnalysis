import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier

def affirmation_train_and_test(owd , path_test, test_comments):

    balance_ratio = 3

    os.chdir(owd)
    cwd = os.getcwd()
    data = pd.read_csv(cwd + '/data/dataframe_train_affirm.csv', encoding = "ISO-8859-1") 
    y = data["affirm"].tolist()
    X_train_raw = data.drop(['sentence', 'open_q', 'close_q', 'reflect'], axis=1)
    
    dict_={}
    data_tr =X_train_raw.reset_index(drop=True)
    aug_data = pd.DataFrame()
    index_ = data_tr.index

    aug_data = data_tr[data_tr['affirm']==1]
    data_tr_bal = data_tr

    for i in range(0,len(data_tr.index)//len(aug_data.index)//balance_ratio):
        data_tr_bal=pd.concat([aug_data,data_tr_bal])
    data_tr_bal =data_tr_bal.reset_index(drop=True)

 
    y_ = data_tr_bal["affirm"].tolist()
    X_ = data_tr_bal.drop(['affirm'], axis=1)
    try:X_=X_.drop(['Unnamed: 0'],axis=1)
    except: pass

    svclassifier = SVC(C=90, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=1, kernel='poly',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    svclassifier.fit(X_, y_)

    adclassifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
                            learning_rate=1, n_estimators=500)
    adclassifier = adclassifier.fit(X_, y_)

    gbclassifier = GradientBoostingClassifier()
    gbclassifier = gbclassifier.fit(X_, y_)


    dtclassifier = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=50,
                       max_features=8, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=None, splitter='best')
    dtclassifier = dtclassifier.fit(X_, y_)

    rfclassifier = RandomForestClassifier(n_estimators=500)
    rfclassifier = rfclassifier.fit(X_, y_)

    etclassifier = ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
                     max_depth=50, max_features=5, max_leaf_nodes=None,
                     min_impurity_decrease=0.0, min_impurity_split=None,
                     min_samples_leaf=1, min_samples_split=2,
                     min_weight_fraction_leaf=0.0, n_estimators=1500,
                     n_jobs=None, oob_score=False, random_state=None, verbose=0,
                     warm_start=False)
    etclassifier = etclassifier.fit(X_, y_)


    data_test = pd.read_csv(path_test) 
    data_test=data_test.drop(['Unnamed: 0'],axis=1)

    y_test = data_test["affirm"].tolist()
    X_test = data_test.drop(['sentence', 'open_q', 'close_q', 'reflect'], axis=1)

    X_test  = X_test.reset_index(drop=True)
    X_test_ = X_test.drop(['affirm'], axis=1)
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
        y_pred_.append(min((ad+et),1))
        
    print(confusion_matrix(y_test,y_pred_))

    token_affirm = ['great' , 'Great' , 'commitment' , 'willingness' ,
                'motivated' , 'motivation' , 'improvements' , 'excellent' , 'Excellent' ,
                'perfect' , 'Perfect' , 'really' , 'shows' , 'nice' , 'Nice', 'fantastic' , 'Fantastic',
                'showing' , 'confidence' , 'well' , 'you\'ve' , 'You\'ve' ,
                'You\'re' , 'you\'re' , 'very', 'deserved' , 'excited' , 'glad' ,
                'happy' , 'appreciate' , 'willing' , 'successes' , 'success' , 'handle' , 'able',
                 'showed' , 'improve' , 'improved' , 'demonstrated' , 'demonstrate' , 'trusted' , 'trust',
                 'presented']
    token_existence = []
    for i in range(len(test_comments)):
        token_existence.append(sum([int(token_ in test_comments[i]) for token_ in token_affirm]) )   



    return confusion_matrix(y_test,y_pred_) , y_pred_, token_existence