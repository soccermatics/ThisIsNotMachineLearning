#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 10:38:39 2021

@author: davsu428
"""

#The task is to learn, whether the lead actor 1 is male or female. Lead actor 2 is tmael
# if actor 1 is female.

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt


import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb


#Student training set
train=pd.read_csv('train.csv')
#Here we load test set with answers
test=pd.read_csv('test_all.csv')
testb=pd.read_csv('test.csv')



all_variables=['Number words female', 'Total words','Number of words lead', 'Difference in words lead and co-lead',
       'Number of male actors', 'Year','Number of female actors', 
       'Number words male', 'Gross','Mean Age Male','Mean Age Female', 'Age Lead', 'Age Co-Lead']

#These variables give lowest AIC as far as I can see.
train_variables=['Number words female', 'Difference in words lead and co-lead',
      'Number of female actors', 'Number of words lead', 'Age Lead', 'Age Co-Lead']
#These variables give lowest AIC as far as I can see.
#train_variables=['Gross','Year']

#train_variables=all_variables

X_train = train[train_variables]
Y_train = train['Lead']
X_test = test[train_variables]
Y_test = test['Lead']
        
model = skl_lm.LogisticRegression(solver='lbfgs')

model.fit(X_train, Y_train)
print('Model summary:')
print(model)

predict_prob = model.predict_proba(X_test)
print('The class order in the model:')
print(model.classes_)
print('Examples of predicted probablities for the above classes:')
predict_prob[0:5]   # inspect the first 5 predictions


prediction = np.empty(len(X_test), dtype=object)
prediction = np.where(predict_prob[:,  0]>=0.5, 'Female', 'Male')
prediction[0:10]  # Inspect the first 5 predictions after labeling.
Y_test[0:10]

# Confusion matrix
print("Confusion matrix:\n")
print(pd.crosstab(prediction, Y_test), '\n')

# Accuracy
print(f"Accuracy: {np.mean(prediction == Y_test):.3f}")



# Number of parameters of logistic regression model
k = model.coef_.size + model.get_params()['fit_intercept']
print(f'The number of parameters is: {k:d}')

# Compute maximum log-likelihood
n = Y_train.size
loglikMale=model.predict_proba(X_train)[:,1]*(Y_train=='Male')
loglikFemale=model.predict_proba(X_train)[:,0]*(Y_train=='Female')
loglik = np.sum(np.log(loglikMale+loglikFemale))
print(f'The log-likelihood is: {loglik:.3f}')

# Compute AIC

AIC = 2 * (k - loglik)
print(f'The AIC is: {AIC:.3f}')

#
#
#misclassification = []
#for k in range(50):  # Try n_neighbors = 1, 2, ...., 50
#    model = skl_nb.KNeighborsClassifier(n_neighbors=k+1)
#    model.fit(X_train, Y_train)
#    prediction = model.predict(X_test)
#    misclassification.append(np.mean(prediction != Y_test))
#
#K = np.linspace(1, 50, 50)
#plt.plot(K, misclassification,'.')
#plt.ylabel('Missclasification')
#plt.xlabel('Number of neighbors')
#plt.show()
