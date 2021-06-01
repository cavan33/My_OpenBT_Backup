#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise: Try classifying the digits dataset with nearest neighbors and
 a linear model. Leave out the last 10% and test prediction performance
 on these observations.
"""
from sklearn import datasets
import numpy as np
X_digits, y_digits = datasets.load_digits(return_X_y=True)
X_digits = X_digits / X_digits.max()
testproportion = round(len(X_digits)*0.1)
X_digits_train = X_digits[:testproportion]
X_digits_test  = X_digits[testproportion:]
y_digits_train = y_digits[:testproportion]
y_digits_test  = y_digits[testproportion:]
# ^ Could be done more elegantly with train_test_split, but my way is OK, too

# With Nearest neighbors:
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_digits_train, y_digits_train)
preds_knn = knn.predict(X_digits_test)
print(preds_knn)
print(y_digits_test)
# Test how many were correct:
print("Percent correct using nearest neighbors method:")
print(sum(preds_knn==y_digits_test)/len(y_digits_test)*100)


# With Linear model:
from sklearn import linear_model
reg = linear_model.SGDClassifier()
reg.fit(X_digits_train, y_digits_train)
preds_lm = reg.predict(X_digits_test)
print(preds_lm)
print(y_digits_test)
# Test how many were correct:
print("Percent correct using linear model method:")
print(sum(preds_lm==y_digits_test)/len(y_digits_test)*100)


# Can compare to the downloaded solution in the tutorial:
# https://scikit-learn.org/stable/tutorial/statistical_inference/supervised_learning.html