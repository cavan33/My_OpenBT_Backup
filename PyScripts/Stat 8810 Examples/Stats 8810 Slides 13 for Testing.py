#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 14:17:11 2021

@author: clark
"""
import numpy as np
import matplotlib.pyplot as plt
# from sklearn import datasets, svm
# Janky importing from openbt-python repo below: (you'll have to change this for your own machine):
import sys
sys.path.append("/home/clark/Documents/OpenBT/openbt-python") # os.getcwd() to check
from openbt2 import OPENBT # I made changes to openbt.py & called it openbt2
     
# CO2 Plume example:
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
# summarize_fitp(fitp13)

fitv13 = m13.vartivity()
# summarize_fitv(fitv13)

fits13 = m13.sobol(cmdopt = 'MPI')
# summarize_fits(fits13)

"""
# Plot the original points and the fit on top:
path = 'Documents/OpenBT/PyScripts/Plots/' # Will be different for your filesystem
from mpl_toolkits.mplot3d import Axes3D
%matplotlib qt5
# ^ Comment this line out if not running in iPython
# To go back to normal plot-showing: Go into Tools-Preferences-Graphics in Spyder, btw
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['axes.labelsize'] = 12;
plt.rcParams['xtick.labelsize'] = 10; plt.rcParams['ytick.labelsize'] = 10;
ax.scatter(co2plume[:,0], co2plume[:,1], co2plume[:,2], color='black')
ax.set_xlabel('Stack_inerts'); ax.set_ylabel('Time'); ax.set_zlabel('CO2')
# plt.savefig(f'{path}co2plume_orig.png')

a = np.arange(0, 1.0001, 1/(npred_arr-1)); b = a;
A, B = np.meshgrid(a, b)
ax.plot_surface(A, B, m13.mmean.reshape(npred_arr,npred_arr), color='black')
ax.set_xlabel('Stack_inerts'); ax.set_ylabel('Time'); ax.set_zlabel('CO2')
plt.savefig(f'{path}co2plume_fit_testing.png')

# Add the uncertainties (keep the surface from above, too):
ax.plot_surface(A, B, (m13.mmean + 1.96 * m13.smean).reshape(npred_arr,npred_arr), color='green')
ax.plot_surface(A, B, (m13.mmean - 1.96 * m13.smean).reshape(npred_arr,npred_arr), color='green')
plt.savefig(f'{path}co2plume_fitp_testing.png')
"""