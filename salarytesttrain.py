# -*- coding: utf-8 -*-
"""
Created on Wed May 20 20:54:01 2020

@author: user
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

Salary_test = pd.read_csv("E:\\Data\\Assignments\\i made\\Naivebayes\\SalaryData_Test.csv")
Salary_train = pd.read_csv("E:\\Data\\Assignments\\i made\\Naivebayes\\SalaryData_Train.csv")

colnames_test = list(Salary_test.columns)
colnames_train = list(Salary_train.columns)

#.......Get Dummies.....
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le = LabelEncoder()

Salary_test['workclass'] = le.fit_transform(Salary_test['workclass'])
Salary_test['education'] = le.fit_transform(Salary_test['education'])
Salary_test['maritalstatus'] = le.fit_transform(Salary_test['maritalstatus'])
Salary_test['occupation'] = le.fit_transform(Salary_test['occupation'])
Salary_test['relationship'] = le.fit_transform(Salary_test['relationship'])
Salary_test['race'] = le.fit_transform(Salary_test['race'])
Salary_test['sex'] = le.fit_transform(Salary_test['sex'])
Salary_test['Salary'] = le.fit_transform(Salary_test['Salary'])
Salary_test['native'] = le.fit_transform(Salary_test['native'])


Salary_train['workclass'] = le.fit_transform(Salary_train['workclass'])
Salary_train['education'] = le.fit_transform(Salary_train['education'])
Salary_train['maritalstatus'] = le.fit_transform(Salary_train['maritalstatus'])
Salary_train['occupation'] = le.fit_transform(Salary_train['occupation'])
Salary_train['relationship'] = le.fit_transform(Salary_train['relationship'])
Salary_train['race'] = le.fit_transform(Salary_train['race'])
Salary_train['sex'] = le.fit_transform(Salary_train['sex'])
Salary_train['Salary'] = le.fit_transform(Salary_train['Salary'])
Salary_train['native'] = le.fit_transform(Salary_train['native'])


# Definig input and output
test_predictors = colnames_test[:13]
test_target = colnames_test[13]

train_predictors = colnames_train[:13]
train_target = colnames_train[13]

# Splitting
DXtrain,DXtest,Dytrain,Dytest = train_test_split(Salary_test[test_predictors],Salary_test[test_target],test_size=0.3, random_state=0)

Dgnb = GaussianNB()
Dmnb = MultinomialNB()

# Prediction
Dpred_gnb = Dgnb.fit(DXtrain,Dytrain).predict(DXtest)
Dpred_mnb = Dmnb.fit(DXtrain,Dytrain).predict(DXtest)

# Confusion Matrix
# GaussianNB
confusion_matrix(Dytest,Dpred_gnb) 
print ("Accuracy",(3223+370)/(3223+370+162+763)) # 79.53

# MultinomialNB
confusion_matrix(Dytest,Dpred_mnb)
print ("Accuracy",(3258+232)/(3258+232+2584+127+901)) # 49.14


DXtrain,DXtest,Dytrain,Dytest = train_test_split(Salary_train[train_predictors],Salary_train[train_target],test_size=0.3, random_state=0)

Dgnb = GaussianNB()
Dmnb = MultinomialNB()

# Prediction

Dpred_gnb = Dgnb.fit(DXtrain,Dytrain).predict(DXtest)
Dpred_mnb = Dmnb.fit(DXtrain,Dytrain).predict(DXtest)

# Confusion Matrix
# GaussianNB
confusion_matrix(Dytest,Dpred_gnb) 
print ("Accuracy",(6463+720)/(6463+720+1531+335)) # 79.37

# MultinomialNB
confusion_matrix(Dytest,Dpred_mnb)
print ("Accuracy",(6526+471)/(6526+471+272+1780)) # 77.32
