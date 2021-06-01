#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 11:20:46 2021

Created to be showcased in my slides on Scikit-Learn Usage; perform a fit on
the Iris data and plot it nicely [very similar to my IrisSVM Challenge from 
the tutorial]
"""

from sklearn import svm, datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Manual pre-processing: split data into 90% train and 10% test subsets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, shuffle=True)

# Pre-processing complete

# Radial Basis Function kernel (default; others are linear and polynomial):
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
# Predicting = determining the class from the training set that best matches
# the X from the testing set
preds_SVC = clf.predict(X_test)
print(preds_SVC)
print(y_test)
# Test how many were correct:
print("Percent correct using RBF SVC method:")
print(round(sum(preds_SVC==y_test)/len(y_test)*100))

