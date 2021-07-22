#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trying to replicate and then squash errors I find on the virtual env from the openbt package
"""
import numpy as np
import matplotlib.pyplot as plt
# from sklearn import datasets, svm
import sys
sys.path.append("openbt-python")
from openbt2 import OPENBT # I made changes to openbt.py & called it openbt2
sys.path.append("PyScripts/Stat 8810 Examples/Functions")
x = [0,0]
y = [1,1]
m = OPENBT(model = "bart")
fit = m.fit(x, y)

from gen_data8810 import *
# Example (Our usual GP realization) originally using BayesTree, 
# now written in Python with openbt-python.
m1 = OPENBT(model = "bart")
design, y1 = gen_data()
fit1 = m.fit(design, y1)

x2 = [[1,1],[2,2]]
y2 = [0,0]
m2 = OPENBT(model = "bart")
fit2 = m.fit(x2, y2)

x3 = np.transpose(np.array([[1,1,1,4], [2,2,5,2]]))
y3 = [1,1,2,2]
m3 = OPENBT(model = "bart")
fit3 = m.fit(x3, y3)