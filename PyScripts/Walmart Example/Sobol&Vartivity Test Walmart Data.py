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
npred_arr = 40000 # Rows of the preds x grid
preds_list = []; np.random.seed(88)
# Categorical variables:
for col in [0, 1, 2, 3]:
     preds_list.append(np.random.randint(np.min(x[:, col]), np.max(x[:, col])+1, size = npred_arr))
# Continuous variables:
for col in [4, 5, 6, 7]:
     preds_list.append(np.random.uniform(np.min(x[:, col]), np.max(x[:, col])+2.2e-16, size = npred_arr))
preds = np.asarray(preds_list).T # This is supposedly faster than having it be a np array the whole time
# print(preds.nbytes / 1000000) # To view storage size in MB

m = OPENBT(model="bart", ndpost=N, nadapt = nadapt, nskip=burn, power=beta,
             base=alpha, tc=tc, numcut=nc, ntree=m, ntreeh=ntreeh, k=k,
             overallsd=shat, overallnu=nu)
fit = m.fit(x,y)
fitp = m.predict(preds)
# summarize_fitp(fitp)

# Vartivity:
fitv = m.vartivity()
# summarize_fitv(fitv)

# Sobol:
fits = m.sobol(cmdopt = 'MPI', tc = 6)
# summarize_fits(fits)

# Save objects:
fpath = '/home/clark/Documents/OpenBT/PyScripts/Walmart Example/Results/'
save_fit_obj(fit, f'{fpath}fit_result.txt', objtype = 'fit')
save_fit_obj(fitp, f'{fpath}fitp_result.txt', objtype = 'fitp')
save_fit_obj(fitv, f'{fpath}fitv_result.txt', objtype = 'fitv')
save_fit_obj(fits, f'{fpath}fits_result.txt', objtype = 'fits')

# Plot y vs yhat plots:
