# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 21:09:32 2024

@author: User
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imbpipeline
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.svm import SVC
import timeit

def plot_lc(pipeline, x_train, x_test, y_train, y_test, title="", png="", seed=1337):
    rand = np.random.RandomState(seed=seed)
    n = len(x_train)
    train_mean_err = []
    cv_mean_err = []
    cv_std_err = []
    test_mean_err = []
    sizes = (np.linspace(.5, 1.0, 10) * n).astype('int')  

    train_times = []
    pred_times = []

    for i in sizes:
        idx = np.random.randint(x_train.shape[0], size=i)
        x_set = x_train[ idx,: ]
        y_set = y_train[ idx ]
        strat_kfold = StratifiedKFold(n_splits=4,shuffle=True, random_state=rand)
        
        scores = cross_validate(pipeline, x_set, y_set, cv=strat_kfold,
                                scoring='accuracy', n_jobs=-1, return_train_score=True)
        
        start = timeit.default_timer()
        pipeline.fit(x_set,y_set)
        end = timeit.default_timer()
        train_times.append(end-start)
        
        train_mean_err.append( np.mean(1-scores['train_score']) ) 
        cv_mean_err.append( np.mean(1-scores['test_score']) ) 
        cv_std_err.append( np.std(1-scores['test_score']) )
        
        start = timeit.default_timer()
        y_pred = pipeline.predict(x_test)
        end = timeit.default_timer()
        pred_times.append(end-start)
        test_mean_err.append( 1-accuracy_score(y_test, y_pred))
    
    train_mean_err = np.array(train_mean_err)
    cv_mean_err = np.array(cv_mean_err)
    cv_std_err = np.array(cv_std_err)
    test_mean_err = np.array(test_mean_err)
    
    train_times = np.array(train_times)
    pred_times = np.array(pred_times)
    
    plt.figure()
    plt.title(title)
    plt.xlabel("Training Samples")
    plt.ylabel("Error Rate")
    plt.plot(sizes, cv_mean_err, 'o-', color="g", label="CV Error")
    plt.plot(sizes, train_mean_err, 'o-', color="b", label="Train Error")
    plt.plot(sizes, test_mean_err, 'o-', color="r", label="Test Error")
    plt.legend()
    plt.savefig(png)
    plt.show()
    return {'Sizes': sizes, 'Test Error': test_mean_err,
            'Train Times': train_times, 'Pred Times': pred_times}

def evaluate(clf, x_train, x_test, y_train, y_test, title="", png=""):
    start = timeit.default_timer()
    clf.fit(x_train, y_train)
    end = timeit.default_timer()
    train_time = end - start

    y_pred = clf.predict(x_test)
    
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test,y_pred)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    cm = confusion_matrix(y_test,y_pred)
    
    print("Model Evaluation")
    print("Train Time (s):   "+"{:.5f}".format(train_time))
    print("F1 Score: %.5f" % f1)
    print("Accuracy: %.5f" % accuracy)
    print("Precision: %.5f" % precision)
    print("Recall: %.5f" % recall)
    print("AUC: %.5f" % auc)
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(title)
    plt.savefig(png)
    plt.show()
    return cm
