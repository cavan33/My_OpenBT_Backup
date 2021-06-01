#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Try classifying classes 1 and 2 from the iris dataset with SVMs, with the 
2 first features. Leave out 10% of each class and test prediction 
performance on these observations.

Warning: the classes are ordered, do not leave out the last 10%, 
you would be testing on only one class.

Hint: You can use the decision_function method on a grid to get intuitions.
"""
from sklearn import svm, datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

X = X[y != 0, :2] # Exclude class 0 and keep only the first two features
y = y[y != 0] # Exclude class 0

# Split data into 90% train and 10% test subsets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, shuffle=True)

# Linear Kernel:
clf = svm.SVC(kernel='linear') # support vector classification
# ^ CLF = classifier. Currently, it's just an instance of an estimator.
# This is a classification problem, by the way.
clf.fit(X_train, y_train)
# Predicting = determining the class from the training set that best matches the X
preds_SVC = clf.predict(X_test)
print(preds_SVC)
print(y_test)
# Test how many were correct:
print("Percent correct using linear SVC method:")
print(sum(preds_SVC==y_test)/len(y_test)*100)

# Polynomial kernel:
clf = svm.SVC(kernel='poly', degree=3)
clf.fit(X_train, y_train)
preds_SVC = clf.predict(X_test)
print(preds_SVC)
print(y_test)
# Test how many were correct:
print("Percent correct using polynomial SVC method:")
print(sum(preds_SVC==y_test)/len(y_test)*100)

# Radial Basis Function kernel:
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
preds_SVC = clf.predict(X_test)
print(preds_SVC)
print(y_test)
# Test how many were correct:
print("Percent correct using RBF SVC method:")
print(sum(preds_SVC==y_test)/len(y_test)*100)

# From multiple tests, all 3 methods seem to get roughly the same amount correct