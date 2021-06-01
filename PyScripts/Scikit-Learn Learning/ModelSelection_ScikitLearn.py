#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 18:36:47 2021

@author: clark
"""
# for the estimator's score_, bigger is better.
from sklearn import datasets, svm
X_digits, y_digits = datasets.load_digits(return_X_y=True)
svc = svm.SVC(C=1, kernel='linear')
svc.fit(X_digits[:-100], y_digits[:-100]).score(X_digits[-100:], y_digits[-100:])

#Split the data into folds, and optimize:
import numpy as np
X_folds = np.array_split(X_digits, 3)
y_folds = np.array_split(y_digits, 3)
scores = list()
for k in range(3):
     # We use 'list' to copy, in order to 'pop' later on
     X_train = list(X_folds)
     X_test = X_train.pop(k)
     X_train = np.concatenate(X_train)
     y_train = list(y_folds)
     y_test = y_train.pop(k)
     y_train = np.concatenate(y_train)
     scores.append(svc.fit(X_train, y_train).score(X_test, y_test))
print(scores)