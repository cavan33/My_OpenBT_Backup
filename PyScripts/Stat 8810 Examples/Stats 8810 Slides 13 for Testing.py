#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 14:17:11 2021

@author: clark
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
# Janky importing from openbt-python repo below: (you'll have to change this for your own machine):
import sys
sys.path.append("~/Documents/OpenBT/openbt-python") # os.getcwd() to check
from openbt2 import OPENBT # I made changes to openbt.py & called it openbt2

def summarize_fitp(fitp):
     print(fitp['mdraws'][0, 0:5]); print(np.mean(fitp['mdraws']))
     print(fitp['sdraws'][0, 0:5]); print(np.mean(fitp['sdraws']))
     # print(fitp['mmean'][0:5]); print(np.mean(fitp['mmean']))
     # print(fitp['smean'][0:5]); print(np.mean(fitp['smean']))
     print(fitp['msd'][0:5]); print(np.mean(fitp['msd']))
     print(fitp['ssd'][0:5]); print(np.mean(fitp['ssd']))

# Settings from further above in the Slides 13 example:
# Tree prior
alpha = 0.95 # Default = 0.95
beta = 2 # Default = 2
# MCMC settings
N = 1000 # (AKA ndpost); Default = 1000
burn = 1000 # (AKA nskip); Default = 100, but we usually use 1000
nadapt = 1000 # Default = 1000
adaptevery = 100 # Default = 100
ntreeh = 1 # Default = 1
tc = 4 # Default = 2, but we usually use 4
npred_arr = 20

# Load data:
co2plume = np.loadtxt('Documents/OpenBT/PyScripts/newco2plume.txt', skiprows=1)
# Kinda cheated, and made the tricky .dat file into a .txt file using R
x = co2plume[:,0:2] # Not including the 3rd column, btw
y = co2plume[:,2]
preds = np.array([(x, y) for x in range(npred_arr) for y in range(npred_arr)])/(npred_arr-1)
preds = np.flip(preds,1) # flipped columns to match the preds in the R code

shat = np.std(y, ddof = 1)
# Try m=200 trees, the recommended default
m=200
# And k=1
k=1
# And nu=1, q=.99
nu=1
# And numcuts=1000
nc=1000

# Do this one manually, since it's a different setup than what I wrote the
# function for:
m13 = OPENBT(model="bart", ndpost=N, nadapt = nadapt, nskip=burn, power=beta,
             base=alpha, tc=tc, numcut=nc, ntree=m, ntreeh=ntreeh, k=k,
             overallsd=shat, overallnu=nu)
fit13 = m13.fit(x,y)
fitp13 = m13.predict(preds)
summarize_fitp(fitp13)



fitv13 = m13.vartivity()
fits13 = m13.sobol(cmdopt = 'MPI')