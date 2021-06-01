#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================
Linear Models (Sec 1.1 of the User Guide)
================================

Going through the examples I find useful (also defining terms in a Word doc)
"""

from sklearn import linear_model
reg = linear_model.LinearRegression()
# LinearRegression(positive=True) if you wanted non-negative coeff's
X = [[0, 0], [1, 1], [2, 2]]
y = [0, 1, 2]
reg.fit(X,y)
reg.coef_
reg.score(X,y)
# See plot_ols.py for a full example

# Ridge Regression: Penalizes large coefficients:
from sklearn import linear_model
reg = linear_model.Ridge(alpha=.5)
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
reg.coef_
reg.intercept_
# Check out RidgeCV later if you need it

# Lasso is useful to decrease the number of nonzero coeff's:
from sklearn import linear_model
reg = linear_model.Lasso(alpha=0.1)
reg.fit([[0, 0], [1, 1]], [0, 1])
reg.predict([[1, 1]])

# The rest of User Guide 1.1 has more niche methods. Reinvestigate these on a 
# case-by-case basis once you need them.