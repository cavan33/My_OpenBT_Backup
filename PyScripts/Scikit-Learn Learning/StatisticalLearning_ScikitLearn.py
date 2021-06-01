#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================
Statistical learning: the setting and
the estimator object in scikit-learn
================================

This section of the tutorial...
"""
from sklearn import datasets
iris=datasets.load_iris()
data=iris.data
data.shape
# Example of reshaping data to (n_samples, n_features)
digits = datasets.load_digits()
digits.images.shape
import matplotlib.pyplot as plt
plt.imshow(digits.images[-1], 
           cmap=plt.cm.gray_r)
data = digits.images.reshape(digits.images.shape[0], -1)
# This can be avoided in the first place with:
# X_digits, y_digits = datasets.load_digits(return_X_y=True)
# X_digits = X_digits / X_digits.max()


# Estimators generic usage:
"""
estimator = Estimator(param1=1, param2=2)
estimator.param1
estimator.estimated_param
"""



# KNN (k nearest neighbors) classification example
# Split iris data in train and test data
# A random permutation, to split the data randomly
import numpy as np
from sklearn import datasets
iris_X, iris_y = datasets.load_iris(return_X_y=True)
np.unique(iris_y)
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]
# Create and fit a nearest-neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)
print(knn.predict(iris_X_test))
print(iris_y_test)
# ^Got all but one correct!



# Linear Regression Usage Example:
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test  = diabetes_X[-20:]
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test  = diabetes_y[-20:]
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
print(regr.coef_)

# The mean square error
np.mean((regr.predict(diabetes_X_test) - diabetes_y_test)**2)

# Explained variance score: 1 is perfect prediction
# and 0 means that there is no linear relationship
# between X and y.
regr.score(diabetes_X_test, diabetes_y_test)

# Ridge and Lasso examples omitted; see tutorial for code. I know them, but only conceptually




# For classification, linear regression is bad because it weights data poorly.
# Solution: Fit a sigmoid function (logistic curve)
log = linear_model.LogisticRegression(C=1e5)
log.fit(iris_X_train, iris_y_train)


# SVMs left out; revisit if you need it later