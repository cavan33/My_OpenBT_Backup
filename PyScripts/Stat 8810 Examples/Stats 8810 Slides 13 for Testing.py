#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 14:17:11 2021

@author: clark
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
# Janky importing from openbt-python repo (you'll have to change this for your own machine):
import sys
sys.path.append("/home/clark/Documents/OpenBT/openbt-python") # os.getcwd() to check
from openbt2 import OPENBT # I made changes to openbt.py & called it openbt2

# Example (Our usual GP realization) originally using BayesTree, 
# now written in Python with openbt-python.

# Set up the function to generate data (originally in dace.sim.r):
def rhogeodacecormat(geoD,rho,alpha=2):
     """
     A function to turn random uniform observations into a data array.
     
     Parameters
     ---------------------
     geoD: an interesting matrix
     rho: IDK
     alpha: specifies depth penalization?
     
     Returns
     -------
     R: Correlation matrix
     """
     # Rho can be an array or number in Python; we'll force it to be an array:
     rho = np.ones(1)*rho
     if (np.any(rho<0)):
          print("rho<0!"); exit()
     if (np.any(rho>1)):
          print("rho>1!"); exit()
     if (np.any(alpha<1) or np.any(alpha>2)):
          print("alpha out of bounds!"); exit()
     if(type(geoD) != np.ndarray):
          print("wrong format for distance matrices list!"); exit()
     # if(len(geoD) != len(rho)):
     #      print("rho vector doesn't match distance list"); exit()
     # ^Got rid of this warning because I'm doing my matrix alg. differently
  
     R = np.ones(shape=(geoD.shape[0], geoD.shape[0])) # Not sure about this line
     for i in range(len(rho)):
          R = R*rho[i]**(geoD**alpha)
          # ^ This is different notation than in R because my geoD array isn't a dataframe
     return(R)


# Generate response (data):
np.random.seed(88)
n = 10; rhotrue = 0.2; lambdatrue = 1
# design = np.random.uniform(size=n).reshape(10,1)           # n x 1
# For testing to compare to R's random seed, use this one:
design = np.array([0.41050128,0.10273570,0.74104481,0.48007870,0.99051343,0.99954223,
          0.03247379,0.76020784,0.67713100,0.97679183]).reshape(n,1)
l1 = np.subtract.outer(design[:,0],design[:,0])            # n x n   
# ^ Not sure about this line, because the m1 is gone
# ^ l1 is the same as l.dez in the R code, by the way
R = rhogeodacecormat(l1,rhotrue)+1e-5*np.diag(np.ones(n))  # n x n
L = np.linalg.cholesky(R)             # n x n
# ^ For some weird reason, I don't need the transpose to make it match up with the L
# matrix in the R code! Took me a while to figure out that one!
# u=np.random.normal(size=R.shape[0]).reshape(1,10)          # n x 1
# For testing to compare to R's random seed, use this one:
u = np.array([0.1237811,0.1331487,-2.0407747,-1.2676089,0.6674839,-0.8014830,
              0.9964860,1.3934232,-0.2291943,0.1707627]).reshape(n,1)
y = np.matmul(L,u)                                         # n x 1

# Now, set up the fit:
# Set values to be used in the fitting/predicting:
# Variance and mean priors
overallsd = 0.2 # (AKA shat) # in reality , it's lambdatrue^2?
# lower shat --> more fitting, I think
overallnu = 3
k = 2 # lower k --> more fitting, I think

# Tree prior
alpha = 0.95 # Default = 0.95
beta = 2 # Default = 2
nc = 10 # (AKA numcut); Default = 100

# MCMC settings
N = 300 # (AKA ndpost); Default = 1000
burn = 10 # (AKA nskip); Default = 100
nadapt = 300 # Default = 1000
tc = 4 # Default = 2
ntree = 1 # Default = 1
ntreeh = 1 # Default = 1

#----------------------------------------------------------------------------------
# Example - the CO2 Plume data from Assignment 3
# Fit the model
co2plume = np.loadtxt('Documents/OpenBT/PyScripts/newco2plume.txt', skiprows=1)
# Kinda cheated, and made it the .dat file into a.txt file using R
x = co2plume[:,0:2] # Not including the 3rd column, btw
y = co2plume[:,2]
preds = np.array([(x, y) for x in range(20) for y in range(20)])/19
preds = np.flip(preds,1) # flipped columns to match the preds in the R code

# Do this one manually, since it's a different setup than what I wrote the
# function for:
m11 = OPENBT(model="bart", ndpost=N, nskip=burn, power=beta, base=alpha,
           tc=tc, numcut=nc, ntree=ntree, ntreeh=ntreeh, k=k,
           overallsd=overallsd, overallnu=overallnu)
fit = m11.fit(x,y)
fitp = m11.predict(preds)

# fitv = m11.vartivity()
fits = m11.sobol(cmdopt = 'MPI')