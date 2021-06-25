#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Import the Walmart data and run sobol/fitv on it to see if there are differences 
(there should be)
"""
import numpy as np
import matplotlib.pyplot as plt
# from sklearn import datasets, svm
# Janky importing from openbt-python repo below: (you'll have to change this for your own machine):
import sys
sys.path.append("/home/clark/Documents/OpenBT/openbt-python") # os.getcwd() to check
from openbt2 import OPENBT
sys.path.append("/home/clark/Documents/OpenBT/PyScripts/Walmart Example")
from Construct_Walmart_Data import *
from summarize_output import *

# Load in the data (8 x variables, after I edited it):
(x, y, x_pd, y_pd) = get_walmart_data()

# Settings:
alpha = 0.95 # Default = 0.95
beta = 2 # Default = 2
# MCMC settings
N = 2000 # (AKA ndpost); Default = 1000
burn = 2000 # (AKA nskip); Default = 100
nadapt = 2000 # Default = 1000
adaptevery = 100 # Default = 100
ntreeh = 1 # Default = 1
tc = 6 # Default = 2, but we usually use 4
shat = np.std(y, ddof = 1)
m=200
k=1
nu=1
nc=2000
npred_arr = 4

preds_grid = {}
for col in range(len(x[0])):
     preds_grid[col] = np.linspace(np.min(x[:, col]), np.max(x[:, col]), num = npred_arr)
# Special case:
preds_grid[3] = np.array([0, 1]) # (Holiday flag can only be 1 or 0)
     
preds = np.array(np.meshgrid(preds_grid[0], preds_grid[1], preds_grid[2], preds_grid[3],
          preds_grid[4], preds_grid[5], preds_grid[6], preds_grid[7], indexing ='xy'
          )).reshape(len(x[0]), npred_arr**(len(x[0])-1)*2).T
# Sort this array to look like the one made in R? For now, skip 
# print(preds[0:10][0:10])

m = OPENBT(model="bart", ndpost=N, nadapt = nadapt, nskip=burn, power=beta,
             base=alpha, tc=tc, numcut=nc, ntree=m, ntreeh=ntreeh, k=k,
             overallsd=shat, overallnu=nu)
fit = m.fit(x,y)
fitp = m.predict(preds)
summarize_fitp(fitp)

# Vartivity:
fitv = m.vartivity()
summarize_fitv(fitv)

# Sobol:
fits = m.sobol(cmdopt = 'MPI', tc = 6)
summarize_fits(fits)
save_fits(fits, '/home/clark/Documents/OpenBT/PyScripts/Walmart Example/fits_result.txt')

"""
# This is probably doomed, but...
# Make plots for each variable vs y (not working yet):
fig = plt.figure(figsize=(10,5.5))
plt.rcParams['axes.labelsize'] = 18; plt.rcParams['axes.titlesize'] = 22;
plt.rcParams['xtick.labelsize'] = 16; plt.rcParams['ytick.labelsize'] = 16;
path = 'Documents/OpenBT/PyScripts/Plots/' # Will be different for your filesystem
x_labels = ["Store", "Month" "Days after 1-1-2010", "Holiday Flag", "Temperature (F)",
            "Fuel Price ($)", "Consumer Price Index", "Unemployment (%)"]
roundnum = [0, 0, 0, 0, 0, 2, 2, 2, 2, 3] # To get it to whole numbers, cents, correct decimals, etc.
preds_grid_rounded = preds_grid
y_predicted = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]}
plt.title(f'Predicted mean Weekly Sales +/- 2 s.d., ntree = {m.ntree}')
   
for i in range(len(x)):
     ax = fig.add_subplot(3, 3, i+1)
     ax.plot(x[:, i], y/1000000, 'ro') # label=current_time can help with plot versions
     ax.set_xlabel(x_labels[i]); ax.set_ylabel('Store Weekly Sales (Millions of $)');
     # plt.legend(loc='upper right')
     # Plot the central line: Overall mean predictor
     # ?

     ax.plot(x[:, i], y_predicted[i]/1000000, 'ro') # label=current_time can help with plot versions
     # Now round everything to actual values for which we can pull the y_predicted:
     
     # Now calculate y_predicted using only the draws that have the correct x-value:
     for 
     y_predicted[i].append(
          np.mean(m.mmean[preds_grid[i][:, i] == preds_grid_rounded[i][;, i]]))
     
     
     
     
     
     ax.plot(preds_grid[i], m.mmean.reshape(npred_arr, len(m.mmean)/npred_arr), 'b-', linewidth=2)
     
     # Make the full plot with the gray lines: (mmeans and smeans are now returned by m.predict()!)
     ax.plot(preds_grid[i], m.mmean - 1.96 * m.smean, color='black', linewidth=0.8)
     ax.plot(preds_grid[i], m.mmean + 1.96 * m.smean, color='black', linewidth=0.8)
     if (ndpost < preds.shape[0]*preds.shape[1]):
          print('Number of posterior draws (ndpost) are less than the number of', 
                'x-values to predict. This is not recommended.')
     for p in range(len(preds_grid[i])):
          ax.plot(preds_grid[i], m.mdraws[p, :],color="gray", linewidth=1, alpha = 0.20)
plt.savefig(f'{path}Walmart_Multiplot.png')
"""