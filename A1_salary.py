# -*- coding: utf-8 -*-
"""
Data from Kaggle
https://www.kaggle.com/datasets/aemyjutt/salary-binary-classifier
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

df_sl = pd.read_csv('salary.csv')
df_sl.describe(include='all')

'''
Preprocessing
'''
df_sl.loc[df_sl['salary'] == ' >50K', 'salary'] = 1
df_sl.loc[df_sl['salary'] == ' <=50K', 'salary'] = 0
df_sl['salary'] = df_sl['salary'].astype('int')

# Drop columns with ?
print(len(df_sl))
for col in df_sl.columns:
    df_sl = df_sl[df_sl[col] != ' ?']
print(len(df_sl))

onehot_cols = ['workclass','fnlwgt','education','marital-status','occupation','relationship','race','sex','native-country']

df_onehot = df_sl[onehot_cols]
df_onehot = pd.get_dummies(df_onehot).astype('int')
df_rest = df_sl.drop(onehot_cols,axis=1)
df_sl = pd.concat([df_rest,df_onehot],axis=1)

x_var = list(df_sl.columns)
x_var.remove('salary')
y_var = ['salary']

print((df_sl['salary'] == 1).sum()/ len(df_sl))

rand = np.random.RandomState(seed=1111)
x_train, x_test, y_train, y_test = train_test_split(np.array(df_sl.loc[:,x_var]),np.array(df_sl.loc[:,y_var]),
                                                    test_size=0.20,random_state=rand)



''' 
Decision Tree
'''
rand = np.random.RandomState(seed=1337)
max_depth_list = [1, 5, 10, 15, 20]
min_samples_list = [1, 3, 5]
dtc = DecisionTreeClassifier(criterion='entropy', random_state=rand)
pipeline = Pipeline(steps = [
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
# {'classifier__max_depth': 10, 'classifier__min_samples_leaf': 5}
print('CV Accuracy: %.5f' % (cv_score*100))
print('Test Accuracy: %.5f' % (test_score*100))
print('CV Error Rate: %.5f' % ((1-cv_score)*100))
print('Test Error Rate: %.5f' % ((1-test_score)*100))


acc_test = []
acc_train = []
max_depths = list(range(1,21))
rand = np.random.RandomState(seed=1337)   
for i in max_depths:
    dtc = DecisionTreeClassifier(max_depth=i, random_state=rand, min_samples_leaf=5, criterion='entropy')
    pipeline = Pipeline(steps = [
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

plt.title('Model Complexity Graph on DTC (Salary)\nVarying Max Tree Depth')
plt.legend()
plt.savefig('2-1-1 MCG DTC Max Tree Depth.png')
plt.show()

acc_test = []
acc_train = []
min_leafs= list(range(1,21))
rand = np.random.RandomState(seed=1337)
for i in min_leafs:       
    dtc = DecisionTreeClassifier(max_depth=10, random_state=rand, min_samples_leaf=i, criterion='entropy')
    pipeline = Pipeline(steps = [
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
plt.title('Model Complexity Graph on DTC (Salary)\nMin Samples Leaf')
plt.legend()
plt.savefig('2-1-2 MCG DTC Min Samples Leaf.png')
plt.show()


rand = np.random.RandomState(seed=1337)
dtc = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, criterion='entropy', random_state=rand)


pipeline = Pipeline(steps = [
    ['scaler', StandardScaler()],
    ['classifier', dtc]
])

DTC_res = plot_lc(
    pipeline, x_train, x_test, y_train, y_test,
    title="Learning Curve on DTC (Salary)\nBest hyperparameters without SMOTE",
    png='2-1-4 LC DTC without SMOTE.png')

evaluate(pipeline, x_train, x_test, y_train, y_test,
         title="Confusion Matrix on DTC (Salary)\nWithout SMOTE",
         png="2-1-6 DTC CM without SMOTE")

'''
Neural Network
'''
rand = np.random.RandomState(seed=1337)
nnc = MLPClassifier(activation='logistic', random_state=rand)
pipeline = Pipeline(steps = [
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
# {'classifier__hidden_layer_sizes': 10, 'classifier__learning_rate_init': 0.001}
print('CV Accuracy: %.5f' % (cv_score*100))
print('Test Accuracy: %.5f' % (test_score*100))
print('CV Error Rate: %.5f' % ((1-cv_score)*100))
print('Test Error Rate: %.5f' % ((1-test_score)*100))

acc_test = []
acc_train = []
hidden_layers = list(range(1,21))
rand = np.random.RandomState(seed=1337)   
for i in hidden_layers:
    nnc = MLPClassifier(hidden_layer_sizes=(i,), learning_rate_init=0.001, activation='logistic', random_state=rand)
    pipeline = Pipeline(steps = [
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

plt.title('Model Complexity Graph on NNC (Salary)\nVarying Hidden Layer Sizes')
plt.legend()
plt.savefig('2-2-1 MCG NNC Hidden Layer Sizes.png')
plt.show()

acc_test = []
acc_train = []
learning_rates= (np.linspace(0.001, 0.1, 20))
rand = np.random.RandomState(seed=1337)
for i in learning_rates:
    nnc = MLPClassifier(hidden_layer_sizes=10, learning_rate_init=i, activation='logistic', random_state=rand)
    pipeline = Pipeline(steps = [
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
plt.title('Model Complexity Graph on NNC (Salary)\nVarying Learning Rate')
plt.legend()
plt.savefig('2-2-2 MCG NNC Learning Rate.png')
plt.show()


rand = np.random.RandomState(seed=1337)
nnc = MLPClassifier(hidden_layer_sizes=10, learning_rate_init=0.001, activation='logistic', random_state=rand)

pipeline = Pipeline(steps = [
    ['scaler', StandardScaler()],
    ['classifier', nnc]
])


NNC_res = plot_lc(
    pipeline, x_train, x_test, y_train, y_test,
    title="Learning Curve on NNC (Salary)\nBest hyperparameters without SMOTE",
    png='2-2-4 LC NNC without SMOTE.png')

evaluate(pipeline, x_train, x_test, y_train, y_test,
         title="Confusion Matrix on NNC (Salary)\nWithout SMOTE",
         png="2-2-6 NNC CM without SMOTE")

''' 
Boosted Decision Tree
'''
rand = np.random.RandomState(seed=1337)
n_estimators_list = [10, 50, 100, 150, 200]
lr_list = [0.001, 0.01, 0.1]
bdc = XGBClassifier(random_state=rand)
pipeline = Pipeline(steps = [
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
    pipeline = Pipeline(steps = [
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

plt.title('Model Complexity Graph on BDC (Salary)\nN Estimators')
plt.legend()
plt.savefig('2-3-1 MCG BDC N Estimators.png')
plt.show()

acc_test = []
acc_train = []
learning_rates= (np.linspace(0.001, 0.1, 10))
rand = np.random.RandomState(seed=1337)   
for i in learning_rates:
    bdc = XGBClassifier(n_estimators=100, learning_rate=i, random_state=rand)
    pipeline = Pipeline(steps = [
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

plt.title('Model Complexity Graph on BDC (Salary)\nLearning Rates')
plt.legend()
plt.savefig('2-3-2 MCG BDC Learning Rates.png')
plt.show()

rand = np.random.RandomState(seed=1337)
bdc = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=rand)

pipeline = Pipeline(steps = [
    ['scaler', StandardScaler()],
    ['classifier', bdc]
])


BDC_res = plot_lc(
    pipeline, x_train, x_test, y_train, y_test,
    title="Learning Curve on BDC (Salary)\nBest hyperparameters without SMOTE",
    png='2-3-4 LC BDC without SMOTE.png')


evaluate(pipeline, x_train, x_test, y_train, y_test,
         title="Confusion Matrix on BDC (Salary)\nWithout SMOTE",
         png="2-3-6 BDC CM without SMOTE")

'''
Support Vector Machine
'''

rand = np.random.RandomState(seed=1337)
kernel_list = ['rbf','linear','sigmoid']
c_list = [0.001, 0.01, 1.0, 10]
svc = SVC(random_state=rand)
pipeline = Pipeline(steps = [
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
#{'classifier__C': 10.0, 'classifier__kernel': 'linear'}
print('CV Accuracy: %.5f' % (cv_score*100))
print('Test Accuracy: %.5f' % (test_score*100))
print('CV Error Rate: %.5f' % ((1-cv_score)*100))
print('Test Error Rate: %.5f' % ((1-test_score)*100))

acc_test = []
acc_train = []
c_list = [0.001, 0.01, 1.0] + list(np.arange(2,21,2).astype('float'))
rand = np.random.RandomState(seed=1337)   
for i in c_list:
    svc = SVC(C=i, kernel='linear', random_state=rand)
    pipeline = Pipeline(steps = [
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

plt.title('Model Complexity Graph on SVC (Salary)\nC')
plt.legend()
plt.savefig('2-4-1 MCG SNC C.png')
plt.show()

rand = np.random.RandomState(seed=1337)
svc = SVC(C=10.0, kernel='linear', random_state=rand)

pipeline = Pipeline(steps = [
    ['scaler', StandardScaler()],
    ['classifier', svc]
])


SVC_res = plot_lc(pipeline, x_train, x_test, y_train, y_test,
        title="Learning Curve on SVC (Salary)\nBest hyperparameters without SMOTE",
        png='2-4-4 LC SVC without SMOTE.png')


evaluate(pipeline, x_train, x_test, y_train, y_test,
         title="Confusion Matrix on SVC (Salary)\nWithout SMOTE",
         png="2-4-6 SVC CM without SMOTE")

'''
K Nearest Neighbours
'''
rand = np.random.RandomState(seed=1337)
k_list = [1,20,40,60,80,100]
leaf_size_list = [10,20,30,40,50]
knc = KNC(n_jobs=-1)
pipeline = Pipeline(steps = [
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
#{'classifier__leaf_size': 10, 'classifier__n_neighbors': 100}
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
    pipeline = Pipeline(steps = [
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

plt.title('Model Complexity Graph on KNC (Salary)\nK Neighbors')
plt.legend()
plt.savefig('2-5-1 MCG KNC K Neighbors.png')
plt.show()

acc_test = []
acc_train = []
leaf_size_list= np.arange(10,210,20).astype('int')
rand = np.random.RandomState(seed=1337)   
for i in leaf_size_list:
    knc = KNC(n_neighbors=100, leaf_size=i, n_jobs=-1)
    pipeline = Pipeline(steps = [
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

plt.title('Model Complexity Graph on KNC (Salary)\nLeaf Size')
plt.legend()
plt.savefig('2-5-2 MCG KNC Leaf Size.png')
plt.show()

rand = np.random.RandomState(seed=1337)
knc = KNC(n_neighbors=100, leaf_size=10, n_jobs=-1)

pipeline = Pipeline(steps = [
    ['scaler', StandardScaler()],
    ['classifier', knc]
])


KNC_res = plot_lc(
    pipeline, x_train, x_test, y_train, y_test,
    title="Learning Curve on KNC (Salary)\nBest hyperparameters without SMOTE",
    png='2-5-4 LC KNC without SMOTE.png')


evaluate(pipeline, x_train, x_test, y_train, y_test,
         title="Confusion Matrix on KNC (Salary)\nWithout SMOTE",
         png="2-5-6 KNC CM without SMOTE")



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
plt.savefig('2-6-2 NonSMOTE Test Error.png')
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
plt.savefig('2-7-2 NonSMOTE Train Times.png')
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
plt.savefig('2-8-2 NonSMOTE Train Times.png')
plt.show()
