#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walmart data, but using a different model (hbart, merck_truncated). The others 
(bt, binomial, etc.) are unfinished on the backend (C commands), so I'll omit those.
4. bart: regular (default model)
5. hbart: works fine, but maybe I should test it again with fully-correct settings
8. merck_truncated: Works (see results/plots)
"""
import numpy as np
import matplotlib.pyplot as plt
# from sklearn import datasets, svm
import sys
sys.path.append("openbt-python") # sys.path to check
from openbt2 import OPENBT
sys.path.append("PyScripts/Walmart Example")
from Construct_Walmart_Data import *
from summarize_output import *
from walmart_pred_plot import *

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
# ntreeh = 1 # Want to use model defaults, but this also works for Merck
tc = 5 # Default = 2
shat = np.std(y, ddof = 1)
m=200
# k=1 # Want to use model defaults, but this also works for Merck
# nu=1 # This also works for Merck
nc=2000

preds_test = x # For in-sample
m_test = OPENBT(model="bart", ndpost=N, nadapt = nadapt, nskip=burn, 
             power=beta, base=alpha, tc=tc, numcut=nc, ntree = m, overallsd=shat)

fit_test = m_test.fit(x,y)
# Now, print the folder where all the config stuff is, so I can inspect:
# print(m_test.fpath)
fitp_test = m_test.predict(preds_test)
# summarize_fitp(fitp_test)

# Vartivity:
fitv_test = m_test.vartivity()
# summarize_fitv(fitv)

# Sobol:
fits_test = m_test.sobol(cmdopt = 'MPI', tc = tc)
# summarize_fits(fits)

# Save fit objects:
fpath1 = 'PyScripts/Results/Walmart/'
save_fit_obj(fit, f'{fpath1}fit_result.txt', objtype = 'fit')
save_fit_obj(fitp, f'{fpath1}fitp_result.txt', objtype = 'fitp')
save_fit_obj(fitv, f'{fpath1}fitv_result.txt', objtype = 'fitv')
save_fit_obj(fits, f'{fpath1}fits_result.txt', objtype = 'fits')


# Next step: See how good the fit is for this "other model" by plotting y vs yhat:     
yhats = fitp_test['mdraws'].mean(axis = 0)
pred_plot(y, yhats, 'In-Sample HBART y vs. $\hat(y)$, Full Settings',
  'PyScripts/Plots/Walmart/Testing_HBART1.png',
  ms = 1.5, millions = True, lims = [0.0, 3.1])