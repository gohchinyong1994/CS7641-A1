# -*- coding: utf-8 -*-
"""
Data from Kaggle
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
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
from sklearn.neighbors import KNeighborsClassifier as KNC
import timeit

os.chdir(r"D:\Georgia Tech\CS7641 Machine Learning\A1")
from A1 import plot_lc, evaluate

pd.set_option('display.max_columns', None)

df_cc = pd.read_csv('creditcard.csv')
df_cc.describe()

print((df_cc['Class'] == 1).sum()/ len(df_cc))

x_var = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

y_var = ['Class']

rand = np.random.RandomState(seed=1111)
x_train, x_test, y_train, y_test = train_test_split(np.array(df_cc.loc[:,x_var]),np.array(df_cc.loc[:,y_var]),
                                                    test_size=0.93,random_state=rand)


''' 
Decision Tree
'''
rand = np.random.RandomState(seed=1337)
max_depth_list = [1, 5, 10, 15, 20]
min_samples_list = [1, 3, 5]
dtc = DecisionTreeClassifier(criterion='entropy', random_state=rand)
pipeline = imbpipeline(steps = [
    ['smote', SMOTE(sampling_strategy='minority',random_state=rand)],
    ['scaler', StandardScaler()],
    ['classifier', dtc]
])
strat_kfold = StratifiedKFold(n_splits=4,shuffle=True, random_state=rand)

dtc_params = {'classifier__max_depth': max_depth_list,
              'classifier__min_samples_leaf': min_samples_list
              }
grid_search_dtc = GridSearchCV(estimator=pipeline, param_grid=dtc_params, scoring='accuracy', cv=strat_kfold, n_jobs=-1, return_train_score=True)
grid_search_dtc.fit(x_train, y_train)

cv_score = grid_search_dtc.best_score_
test_score = grid_search_dtc.score(x_test, y_test)
print('Grid Search DTC')
print(grid_search_dtc.best_params_)
print('CV Accuracy: %.5f' % (cv_score*100))
print('Test Accuracy: %.5f' % (test_score*100))
print('CV Error Rate: %.5f' % ((1-cv_score)*100))
print('Test Error Rate: %.5f' % ((1-test_score)*100))


acc_test = []
acc_train = []
max_depths = list(range(1,21))
rand = np.random.RandomState(seed=1337)   
for i in max_depths:
    dtc = DecisionTreeClassifier(max_depth=i, random_state=rand, min_samples_leaf=1, criterion='entropy')
    pipeline = imbpipeline(steps = [
        ['smote', SMOTE(sampling_strategy='minority',random_state=rand)],
        ['scaler', StandardScaler()],
        ['classifier', dtc]
    ])
    pipeline.fit(x_train, y_train)
    y_pred_test = pipeline.predict(x_test)
    y_pred_train = pipeline.predict(x_train)
    acc_test.append(accuracy_score(y_test, y_pred_test))
    acc_train.append(accuracy_score(y_train, y_pred_train))
  
err_test = [1.0 - x for x in acc_test]
err_train = [1.0 - x for x in acc_train]
plt.figure()
plt.plot(max_depths, err_test,'.-' ,color='r', label='Test Error')
plt.plot(max_depths, err_train,'.-',color='b', label='Train Error')
plt.ylabel('Error Rate')
plt.xlabel('Max Tree Depth')

plt.title('Model Complexity Graph on DTC (Credit Card Fraud)\nVarying Max Tree Depth')
plt.legend()
plt.savefig('1-1-1 MCG DTC Max Tree Depth.png')
plt.show()

acc_test = []
acc_train = []
min_leafs= list(range(1,21))
rand = np.random.RandomState(seed=1337)
for i in min_leafs:       
    dtc = DecisionTreeClassifier(max_depth=10, random_state=rand, min_samples_leaf=i, criterion='entropy')
    pipeline = imbpipeline(steps = [
        ['smote', SMOTE(sampling_strategy='minority',random_state=rand)],
        ['scaler', StandardScaler()],
        ['classifier', dtc]
    ])
    pipeline.fit(x_train, y_train)
    y_pred_test = pipeline.predict(x_test)
    y_pred_train = pipeline.predict(x_train)
    acc_test.append(accuracy_score(y_test, y_pred_test))
    acc_train.append(accuracy_score(y_train, y_pred_train))
  
err_test = [1.0 - x for x in acc_test]
err_train = [1.0 - x for x in acc_train]
plt.figure()
plt.plot(min_leafs, err_test,'.-' ,color='r', label='Test Error')
plt.plot(min_leafs, err_train,'.-',color='b', label='Train Error')
plt.ylabel('Error Rate')
plt.xlabel('Min Samples Leaf')
plt.title('Model Complexity Graph on DTC (Credit Card Fraud)\nMin Samples Leaf')
plt.legend()
plt.savefig('1-1-2 MCG DTC Min Samples Leaf.png')
plt.show()


rand = np.random.RandomState(seed=1337)
dtc = DecisionTreeClassifier(max_depth=10, min_samples_leaf=1, criterion='entropy', random_state=rand)

pipeline_smote = imbpipeline(steps = [
    ['smote', SMOTE(sampling_strategy='minority',random_state=rand)],
    ['scaler', StandardScaler()],
    ['classifier', dtc]
])

pipeline = Pipeline(steps = [
    ['scaler', StandardScaler()],
    ['classifier', dtc]
])

DTC_SMOTE_res = plot_lc(
    pipeline_smote, x_train, x_test, y_train, y_test,
    title="Learning Curve on DTC (Credit Card Fraud)\nBest hyperparameters with SMOTE",
    png='1-1-3 LC DTC with SMOTE.png')

DTC_res = plot_lc(
    pipeline, x_train, x_test, y_train, y_test,
    title="Learning Curve on DTC (Credit Card Fraud)\nBest hyperparameters without SMOTE",
    png='1-1-4 LC DTC without SMOTE.png')

evaluate(pipeline_smote, x_train, x_test, y_train, y_test,
         title="Confusion Matrix on DTC (Credit Card Fraud)\nWith SMOTE",
         png="1-1-5 DTC CM with SMOTE")

evaluate(pipeline, x_train, x_test, y_train, y_test,
         title="Confusion Matrix on DTC (Credit Card Fraud)\nWithout SMOTE",
         png="1-1-6 DTC CM without SMOTE")

'''
Neural Network
'''
rand = np.random.RandomState(seed=1337)
nnc = MLPClassifier(activation='logistic', random_state=rand)
pipeline = imbpipeline(steps = [
    ['smote', SMOTE(sampling_strategy='minority',random_state=rand)],
    ['scaler', StandardScaler()],
    ['classifier', nnc]
])
strat_kfold = StratifiedKFold(n_splits=4,shuffle=True, random_state=rand)

hidden_layers_list = [1,5,10,15,20]
lr_list = [0.001, 0.01, 0.1]
nnc_params = {'classifier__learning_rate_init': lr_list, 'classifier__hidden_layer_sizes': hidden_layers_list, }

grid_search_nnc = GridSearchCV(estimator=pipeline, param_grid=nnc_params, scoring='accuracy', cv=strat_kfold, n_jobs=-1, return_train_score=True)
grid_search_nnc.fit(x_train, y_train)

cv_score = grid_search_nnc.best_score_
test_score = grid_search_nnc.score(x_test, y_test)
print('Grid Search NNC')
print(grid_search_nnc.best_params_)
# {'classifier__hidden_layer_sizes': 15, 'classifier__learning_rate_init': 0.1}
print('CV Accuracy: %.5f' % (cv_score*100))
print('Test Accuracy: %.5f' % (test_score*100))
print('CV Error Rate: %.5f' % ((1-cv_score)*100))
print('Test Error Rate: %.5f' % ((1-test_score)*100))

acc_test = []
acc_train = []
hidden_layers = list(range(1,21))
rand = np.random.RandomState(seed=1337)   
for i in hidden_layers:
    nnc = MLPClassifier(hidden_layer_sizes=(i,), learning_rate_init=0.1, activation='logistic', random_state=rand)
    pipeline = imbpipeline(steps = [
        ['smote', SMOTE(sampling_strategy='minority',random_state=rand)],
        ['scaler', StandardScaler()],
        ['classifier', nnc]
    ])
    pipeline.fit(x_train, y_train)
    y_pred_test = pipeline.predict(x_test)
    y_pred_train = pipeline.predict(x_train)
    acc_test.append(accuracy_score(y_test, y_pred_test))
    acc_train.append(accuracy_score(y_train, y_pred_train))
  
err_test = [1.0 - x for x in acc_test]
err_train = [1.0 - x for x in acc_train]
plt.figure()
plt.plot(hidden_layers, err_test,'.-' ,color='r', label='Test Error')
plt.plot(hidden_layers, err_train,'.-',color='b', label='Train Error')
plt.ylabel('Error Rate')
plt.xlabel('Hidden Layer Sizes')

plt.title('Model Complexity Graph on NNC (Credit Card Fraud)\nVarying Hidden Layer Sizes')
plt.legend()
plt.savefig('1-2-1 MCG NNC Hidden Layer Sizes.png')
plt.show()

acc_test = []
acc_train = []
learning_rates= (np.linspace(0.001, 0.1, 20))
rand = np.random.RandomState(seed=1337)
for i in learning_rates:
    nnc = MLPClassifier(hidden_layer_sizes=15, learning_rate_init=i, activation='logistic', random_state=rand)
    pipeline = imbpipeline(steps = [
        ['smote', SMOTE(sampling_strategy='minority',random_state=rand)],
        ['scaler', StandardScaler()],
        ['classifier', nnc]
    ])
    pipeline.fit(x_train, y_train)
    y_pred_test = pipeline.predict(x_test)
    y_pred_train = pipeline.predict(x_train)
    acc_test.append(accuracy_score(y_test, y_pred_test))
    acc_train.append(accuracy_score(y_train, y_pred_train))
  
err_test = [1.0 - x for x in acc_test]
err_train = [1.0 - x for x in acc_train]
plt.figure()
plt.plot(learning_rates, err_test,'.-' ,color='r', label='Test Error')
plt.plot(learning_rates, err_train,'.-',color='b', label='Train Error')
plt.ylabel('Error Rate')
plt.xlabel('Learning Rates')
plt.title('Model Complexity Graph on NNC (Credit Card Fraud)\nVarying Learning Rate')
plt.legend()
plt.savefig('1-2-2 MCG NNC Learning Rate.png')
plt.show()


rand = np.random.RandomState(seed=1337)
nnc = MLPClassifier(hidden_layer_sizes=15, learning_rate_init=0.1, activation='logistic', random_state=rand)
pipeline_smote = imbpipeline(steps = [
    ['smote', SMOTE(sampling_strategy='minority',random_state=rand)],
    ['scaler', StandardScaler()],
    ['classifier', nnc]
])

pipeline = Pipeline(steps = [
    ['scaler', StandardScaler()],
    ['classifier', nnc]
])

NNC_SMOTE_res = plot_lc(
    pipeline_smote, x_train, x_test, y_train, y_test,
    title="Learning Curve on NNC (Credit Card Fraud)\nBest hyperparameters with SMOTE",
    png='1-2-3 LC NNC with SMOTE.png')

NNC_res = plot_lc(
    pipeline, x_train, x_test, y_train, y_test,
    title="Learning Curve on NNC (Credit Card Fraud)\nBest hyperparameters without SMOTE",
    png='1-2-4 LC NNC without SMOTE.png')

evaluate(pipeline_smote, x_train, x_test, y_train, y_test,
         title="Confusion Matrix on NNC (Credit Card Fraud)\nWith SMOTE",
         png="1-2-5 NNC CM with SMOTE")

evaluate(pipeline, x_train, x_test, y_train, y_test,
         title="Confusion Matrix on NNC (Credit Card Fraud)\nWithout SMOTE",
         png="1-2-6 NNC CM without SMOTE")

''' 
Boosted Decision Tree
'''
rand = np.random.RandomState(seed=1337)
n_estimators_list = [10, 50, 100, 150, 200]
lr_list = [0.001, 0.01, 0.1]
bdc = XGBClassifier(random_state=rand)
pipeline = imbpipeline(steps = [
    ['smote', SMOTE(sampling_strategy='minority',random_state=rand)],
    ['scaler', StandardScaler()],
    ['classifier', bdc]
])
strat_kfold = StratifiedKFold(n_splits=4,shuffle=True, random_state=rand)

bdc_params = {#'classifier__max_depth': max_depth_list,
              #'classifier__min_samples_leaf': min_samples_list,
              'classifier__n_estimators': n_estimators_list,
              'classifier__learning_rate': lr_list,
              }
grid_search_bdc = GridSearchCV(estimator=pipeline, param_grid=bdc_params, scoring='accuracy', cv=strat_kfold, n_jobs=-1, return_train_score=True)
grid_search_bdc.fit(x_train, y_train)

cv_score = grid_search_bdc.best_score_
test_score = grid_search_bdc.score(x_test, y_test)
print('Grid Search BDC')
print(grid_search_bdc.best_params_)
#{'classifier__learning_rate': 0.1, 'classifier__n_estimators': 100}
print('CV Accuracy: %.5f' % (cv_score*100))
print('Test Accuracy: %.5f' % (test_score*100))
print('CV Error Rate: %.5f' % ((1-cv_score)*100))
print('Test Error Rate: %.5f' % ((1-test_score)*100))


acc_test = []
acc_train = []
n_estimators = np.linspace(10, 200, 10).astype('int')
rand = np.random.RandomState(seed=1337)   
for i in n_estimators:
    bdc = XGBClassifier(n_estimators=i, learning_rate=0.1, random_state=rand)
    pipeline = imbpipeline(steps = [
        ['smote', SMOTE(sampling_strategy='minority',random_state=rand)],
        ['scaler', StandardScaler()],
        ['classifier', bdc]
    ])
    pipeline.fit(x_train, y_train)
    y_pred_test = pipeline.predict(x_test)
    y_pred_train = pipeline.predict(x_train)
    acc_test.append(accuracy_score(y_test, y_pred_test))
    acc_train.append(accuracy_score(y_train, y_pred_train))
  
err_test = [1.0 - x for x in acc_test]
err_train = [1.0 - x for x in acc_train]
plt.figure()
plt.plot(n_estimators, err_test,'.-' ,color='r', label='Test Error')
plt.plot(n_estimators, err_train,'.-',color='b', label='Train Error')
plt.ylabel('Error Rate')
plt.xlabel('N Estimators')

plt.title('Model Complexity Graph on BDC (Credit Card Fraud)\nN Estimators')
plt.legend()
plt.savefig('1-3-1 MCG BDC N Estimators.png')
plt.show()

acc_test = []
acc_train = []
learning_rates= (np.linspace(0.001, 0.1, 10))
rand = np.random.RandomState(seed=1337)   
for i in learning_rates:
    bdc = XGBClassifier(n_estimators=100, learning_rate=i, random_state=rand)
    pipeline = imbpipeline(steps = [
        ['smote', SMOTE(sampling_strategy='minority',random_state=rand)],
        ['scaler', StandardScaler()],
        ['classifier', bdc]
    ])
    pipeline.fit(x_train, y_train)
    y_pred_test = pipeline.predict(x_test)
    y_pred_train = pipeline.predict(x_train)
    acc_test.append(accuracy_score(y_test, y_pred_test))
    acc_train.append(accuracy_score(y_train, y_pred_train))
  
err_test = [1.0 - x for x in acc_test]
err_train = [1.0 - x for x in acc_train]
plt.figure()
plt.plot(learning_rates, err_test,'.-' ,color='r', label='Test Error')
plt.plot(learning_rates, err_train,'.-',color='b', label='Train Error')
plt.ylabel('Error Rate')
plt.xlabel('Learning Rates')

plt.title('Model Complexity Graph on BDC (Credit Card Fraud)\nLearning Rates')
plt.legend()
plt.savefig('1-3-2 MCG BDC Learning Rates.png')
plt.show()

rand = np.random.RandomState(seed=1337)
bdc = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=rand)
pipeline_smote = imbpipeline(steps = [
    ['smote', SMOTE(sampling_strategy='minority',random_state=rand)],
    ['scaler', StandardScaler()],
    ['classifier', bdc]
])

pipeline = Pipeline(steps = [
    ['scaler', StandardScaler()],
    ['classifier', bdc]
])

BDC_SMOTE_res = plot_lc(
    pipeline_smote, x_train, x_test, y_train, y_test,
    title="Learning Curve on BDC (Credit Card Fraud)\nBest hyperparameters with SMOTE",
    png='1-3-3 LC BDC with SMOTE.png')

BDC_res = plot_lc(
    pipeline, x_train, x_test, y_train, y_test,
    title="Learning Curve on BDC (Credit Card Fraud)\nBest hyperparameters without SMOTE",
    png='1-3-4 LC BDC without SMOTE.png')

evaluate(pipeline_smote, x_train, x_test, y_train, y_test,
         title="Confusion Matrix on BDC (Credit Card Fraud)\nWith SMOTE",
         png="1-3-5 BDC CM with SMOTE")

evaluate(pipeline, x_train, x_test, y_train, y_test,
         title="Confusion Matrix on BDC (Credit Card Fraud)\nWithout SMOTE",
         png="1-3-6 BDC CM without SMOTE")

'''
Support Vector Machine
'''

rand = np.random.RandomState(seed=1337)
kernel_list = ['rbf','linear','sigmoid']
c_list = [0.001, 0.01, 1.0, 10]
svc = SVC(random_state=rand)
pipeline = imbpipeline(steps = [
    ['smote', SMOTE(sampling_strategy='minority',random_state=rand)],
    ['scaler', StandardScaler()],
    ['classifier', svc]
])
strat_kfold = StratifiedKFold(n_splits=4,shuffle=True, random_state=rand)

svc_params = {'classifier__kernel': kernel_list,
              'classifier__C': [10.0],
              }
grid_search_svc = GridSearchCV(estimator=pipeline, param_grid=svc_params, scoring='accuracy', cv=strat_kfold, n_jobs=-1, return_train_score=True)
grid_search_svc.fit(x_train, y_train)

cv_score = grid_search_svc.best_score_
test_score = grid_search_svc.score(x_test, y_test)
print('Grid Search SVC')
print(grid_search_svc.best_params_)
#{'classifier__C': 10.0, 'classifier__kernel': 'rbf'}
print('CV Accuracy: %.5f' % (cv_score*100))
print('Test Accuracy: %.5f' % (test_score*100))
print('CV Error Rate: %.5f' % ((1-cv_score)*100))
print('Test Error Rate: %.5f' % ((1-test_score)*100))

acc_test = []
acc_train = []
c_list = [0.001, 0.01, 1.0] + list(np.arange(2,21,2).astype('float'))
rand = np.random.RandomState(seed=1337)   
for i in c_list:
    svc = SVC(C=i, kernel='rbf', random_state=rand)
    pipeline = imbpipeline(steps = [
        ['smote', SMOTE(sampling_strategy='minority',random_state=rand)],
        ['scaler', StandardScaler()],
        ['classifier', svc]
    ])
    pipeline.fit(x_train, y_train)
    y_pred_test = pipeline.predict(x_test)
    y_pred_train = pipeline.predict(x_train)
    acc_test.append(accuracy_score(y_test, y_pred_test))
    acc_train.append(accuracy_score(y_train, y_pred_train))
  
err_test = [1.0 - x for x in acc_test]
err_train = [1.0 - x for x in acc_train]
plt.figure()
plt.plot(c_list, err_test,'.-' ,color='r', label='Test Error')
plt.plot(c_list, err_train,'.-',color='b', label='Train Error')
plt.ylabel('Error Rate')
plt.xlabel('C')

plt.title('Model Complexity Graph on SVC (Credit Card Fraud)\nC')
plt.legend()
plt.savefig('1-4-1 MCG SNC C.png')
plt.show()

rand = np.random.RandomState(seed=1337)
svc = SVC(C=10.0, kernel='rbf', random_state=rand)
pipeline_smote = imbpipeline(steps = [
    ['smote', SMOTE(sampling_strategy='minority',random_state=rand)],
    ['scaler', StandardScaler()],
    ['classifier', svc]
])

pipeline = Pipeline(steps = [
    ['scaler', StandardScaler()],
    ['classifier', svc]
])

SVC_SMOTE_res = plot_lc(pipeline_smote, x_train, x_test, y_train, y_test,
        title="Learning Curve on SVC (Credit Card Fraud)\nBest hyperparameters with SMOTE",
        png='1-4-3 LC SVC with SMOTE.png')

SVC_res = plot_lc(pipeline, x_train, x_test, y_train, y_test,
        title="Learning Curve on SVC (Credit Card Fraud)\nBest hyperparameters without SMOTE",
        png='1-4-4 LC SVC without SMOTE.png')

evaluate(pipeline_smote, x_train, x_test, y_train, y_test,
         title="Confusion Matrix on SVC (Credit Card Fraud)\nWith SMOTE",
         png="1-4-5 SVC CM with SMOTE")

evaluate(pipeline, x_train, x_test, y_train, y_test,
         title="Confusion Matrix on SVC (Credit Card Fraud)\nWithout SMOTE",
         png="1-4-6 SVC CM without SMOTE")

'''
K Nearest Neighbours
'''
rand = np.random.RandomState(seed=1337)
k_list = [1,20,40,60,80,100]
leaf_size_list = [10,20,30,40,50]
knc = KNC(n_jobs=-1)
pipeline = imbpipeline(steps = [
    ['smote', SMOTE(sampling_strategy='minority',random_state=rand)],
    ['scaler', StandardScaler()],
    ['classifier', knc]
])
strat_kfold = StratifiedKFold(n_splits=4,shuffle=True, random_state=rand)

knc_params = {'classifier__n_neighbors': k_list,
              'classifier__leaf_size': leaf_size_list,
              }
grid_search_knc = GridSearchCV(estimator=pipeline, param_grid=knc_params, scoring='accuracy', cv=strat_kfold, n_jobs=-1, return_train_score=True)
grid_search_knc.fit(x_train, y_train)

cv_score = grid_search_knc.best_score_
test_score = grid_search_knc.score(x_test, y_test)
print('Grid Search KNC')
print(grid_search_knc.best_params_)
#{'classifier__leaf_size': 10, 'classifier__n_neighbors': 1}
print('CV Accuracy: %.5f' % (cv_score*100))
print('Test Accuracy: %.5f' % (test_score*100))
print('CV Error Rate: %.5f' % ((1-cv_score)*100))
print('Test Error Rate: %.5f' % ((1-test_score)*100))


acc_test = []
acc_train = []
k_list = np.linspace(1, 200, 10).astype('int')
rand = np.random.RandomState(seed=1337)   
for i in k_list:
    knc = KNC(n_neighbors=i, leaf_size=10, n_jobs=-1)
    pipeline = imbpipeline(steps = [
        ['smote', SMOTE(sampling_strategy='minority',random_state=rand)],
        ['scaler', StandardScaler()],
        ['classifier', knc]
    ])
    pipeline.fit(x_train, y_train)
    y_pred_test = pipeline.predict(x_test)
    y_pred_train = pipeline.predict(x_train)
    acc_test.append(accuracy_score(y_test, y_pred_test))
    acc_train.append(accuracy_score(y_train, y_pred_train))
  
err_test = [1.0 - x for x in acc_test]
err_train = [1.0 - x for x in acc_train]
plt.figure()
plt.plot(k_list, err_test,'.-' ,color='r', label='Test Error')
plt.plot(k_list, err_train,'.-',color='b', label='Train Error')
plt.ylabel('Error Rate')
plt.xlabel('K Neighbors')

plt.title('Model Complexity Graph on KNC (Credit Card Fraud)\nK Neighbors')
plt.legend()
plt.savefig('1-5-1 MCG KNC K Neighbors.png')
plt.show()

acc_test = []
acc_train = []
leaf_size_list= np.arange(10,210,20).astype('int')
rand = np.random.RandomState(seed=1337)   
for i in leaf_size_list:
    knc = KNC(n_neighbors=1, leaf_size=i, n_jobs=-1)
    pipeline = imbpipeline(steps = [
        ['smote', SMOTE(sampling_strategy='minority',random_state=rand)],
        ['scaler', StandardScaler()],
        ['classifier', knc]
    ])
    pipeline.fit(x_train, y_train)
    y_pred_test = pipeline.predict(x_test)
    y_pred_train = pipeline.predict(x_train)
    acc_test.append(accuracy_score(y_test, y_pred_test))
    acc_train.append(accuracy_score(y_train, y_pred_train))
  
err_test = [1.0 - x for x in acc_test]
err_train = [1.0 - x for x in acc_train]
plt.figure()
plt.plot(leaf_size_list, err_test,'.-' ,color='r', label='Test Error')
plt.plot(leaf_size_list, err_train,'.-',color='b', label='Train Error')
plt.ylabel('Error Rate')
plt.xlabel('Leaf Size')

plt.title('Model Complexity Graph on KNC (Credit Card Fraud)\nLeaf Size')
plt.legend()
plt.savefig('1-5-2 MCG KNC Leaf Size.png')
plt.show()

rand = np.random.RandomState(seed=1337)
knc = KNC(n_neighbors=1, leaf_size=10, n_jobs=-1)
pipeline_smote = imbpipeline(steps = [
    ['smote', SMOTE(sampling_strategy='minority',random_state=rand)],
    ['scaler', StandardScaler()],
    ['classifier', knc]
])

pipeline = Pipeline(steps = [
    ['scaler', StandardScaler()],
    ['classifier', knc]
])

KNC_SMOTE_res = plot_lc(
    pipeline_smote, x_train, x_test, y_train, y_test,
    title="Learning Curve on KNC (Credit Card Fraud)\nBest hyperparameters with SMOTE",
    png='1-5-3 LC KNC with SMOTE.png')

KNC_res = plot_lc(
    pipeline, x_train, x_test, y_train, y_test,
    title="Learning Curve on KNC (Credit Card Fraud)\nBest hyperparameters without SMOTE",
    png='1-5-4 LC KNC without SMOTE.png')

evaluate(pipeline_smote, x_train, x_test, y_train, y_test,
         title="Confusion Matrix on KNC (Credit Card Fraud)\nWith SMOTE",
         png="1-5-5 KNC CM with SMOTE")

evaluate(pipeline, x_train, x_test, y_train, y_test,
         title="Confusion Matrix on KNC (Credit Card Fraud)\nWithout SMOTE",
         png="1-5-6 KNC CM without SMOTE")

res_SMOTE_map = {
    'DTC': DTC_SMOTE_res,
    'NNC': NNC_SMOTE_res,
    'BDC': BDC_SMOTE_res,
    'SVC': SVC_SMOTE_res,
    'KNC': KNC_SMOTE_res,
    }

res_map = {
    'DTC': DTC_res,
    'NNC': NNC_res,
    'BDC': BDC_res,
    'SVC': SVC_res,
    'KNC': KNC_res,
    }


plt.figure()
plt.plot(DTC_SMOTE_res['Sizes'], DTC_SMOTE_res['Test Error'],'.-' ,color='r', label='DTC')
plt.plot(NNC_SMOTE_res['Sizes'], NNC_SMOTE_res['Test Error'],'.-' ,color='g', label='NNC')
plt.plot(BDC_SMOTE_res['Sizes'], BDC_SMOTE_res['Test Error'],'.-' ,color='b', label='BDC')
plt.plot(SVC_SMOTE_res['Sizes'], SVC_SMOTE_res['Test Error'],'.-' ,color='y', label='SVC')
plt.plot(KNC_SMOTE_res['Sizes'], KNC_SMOTE_res['Test Error'],'.-' ,color='c', label='KNC')
plt.ylabel('Error Rate')
plt.xlabel('Training Samples')

plt.title('Test Error v Learning Size (Credit Card Fraud)\nComparison between SMOTE algorithms')
plt.legend()
plt.savefig('1-6-1 SMOTE Test Error.png')
plt.show()

plt.figure()
plt.plot(DTC_res['Sizes'], DTC_res['Test Error'],'.-' ,color='r', label='DTC')
plt.plot(NNC_res['Sizes'], NNC_res['Test Error'],'.-' ,color='g', label='NNC')
plt.plot(BDC_res['Sizes'], BDC_res['Test Error'],'.-' ,color='b', label='BDC')
plt.plot(SVC_res['Sizes'], SVC_res['Test Error'],'.-' ,color='y', label='SVC')
plt.plot(KNC_res['Sizes'], KNC_res['Test Error'],'.-' ,color='c', label='KNC')
plt.ylabel('Error Rate')
plt.xlabel('Training Samples')

plt.title('Test Error v Learning Size (Credit Card Fraud)\nComparison between non-SMOTE algorithms')
plt.legend()
plt.savefig('1-6-2 NonSMOTE Test Error.png')
plt.show()


plt.figure()
plt.plot(DTC_SMOTE_res['Sizes'], DTC_SMOTE_res['Train Times'],'.-' ,color='r', label='DTC')
plt.plot(NNC_SMOTE_res['Sizes'], NNC_SMOTE_res['Train Times'],'.-' ,color='g', label='NNC')
plt.plot(BDC_SMOTE_res['Sizes'], BDC_SMOTE_res['Train Times'],'.-' ,color='b', label='BDC')
plt.plot(SVC_SMOTE_res['Sizes'], SVC_SMOTE_res['Train Times'],'.-' ,color='y', label='SVC')
plt.plot(KNC_SMOTE_res['Sizes'], KNC_SMOTE_res['Train Times'],'.-' ,color='c', label='KNC')
plt.ylabel('Train Time (s)')
plt.xlabel('Training Samples')

plt.title('Train Times v Learning Size (Credit Card Fraud)\nComparison between SMOTE algorithms')
plt.legend()
plt.savefig('1-7-1 SMOTE Train Times.png')
plt.show()

plt.figure()
plt.plot(DTC_res['Sizes'], DTC_res['Train Times'],'.-' ,color='r', label='DTC')
plt.plot(NNC_res['Sizes'], NNC_res['Train Times'],'.-' ,color='g', label='NNC')
plt.plot(BDC_res['Sizes'], BDC_res['Train Times'],'.-' ,color='b', label='BDC')
plt.plot(SVC_res['Sizes'], SVC_res['Train Times'],'.-' ,color='y', label='SVC')
plt.plot(KNC_res['Sizes'], KNC_res['Train Times'],'.-' ,color='c', label='KNC')
plt.ylabel('Train Time (s)')
plt.xlabel('Training Samples')

plt.title('Train Times v Learning Size (Credit Card Fraud)\nComparison between non-SMOTE algorithms')
plt.legend()
plt.savefig('1-7-2 NonSMOTE Train Times.png')
plt.show()


plt.figure()
plt.plot(DTC_SMOTE_res['Sizes'], DTC_SMOTE_res['Pred Times'],'.-' ,color='r', label='DTC')
plt.plot(NNC_SMOTE_res['Sizes'], NNC_SMOTE_res['Pred Times'],'.-' ,color='g', label='NNC')
plt.plot(BDC_SMOTE_res['Sizes'], BDC_SMOTE_res['Pred Times'],'.-' ,color='b', label='BDC')
plt.plot(SVC_SMOTE_res['Sizes'], SVC_SMOTE_res['Pred Times'],'.-' ,color='y', label='SVC')
plt.plot(KNC_SMOTE_res['Sizes'], KNC_SMOTE_res['Pred Times'],'.-' ,color='c', label='KNC')
plt.ylabel('Pred Time (s)')
plt.xlabel('Training Samples')

plt.title('Pred Times v Learning Size (Credit Card Fraud)\nComparison between SMOTE algorithms')
plt.legend()
plt.savefig('1-8-1 SMOTE Train Times.png')
plt.show()

plt.figure()
plt.plot(DTC_res['Sizes'], DTC_res['Pred Times'],'.-' ,color='r', label='DTC')
plt.plot(NNC_res['Sizes'], NNC_res['Pred Times'],'.-' ,color='g', label='NNC')
plt.plot(BDC_res['Sizes'], BDC_res['Pred Times'],'.-' ,color='b', label='BDC')
plt.plot(SVC_res['Sizes'], SVC_res['Pred Times'],'.-' ,color='y', label='SVC')
plt.plot(KNC_res['Sizes'], KNC_res['Pred Times'],'.-' ,color='c', label='KNC')
plt.ylabel('Pred Time (s)')
plt.xlabel('Training Samples')

plt.title('Pred Times v Learning Size (Credit Card Fraud)\nComparison between non-SMOTE algorithms')
plt.legend()
plt.savefig('1-8-2 NonSMOTE Train Times.png')
plt.show()
