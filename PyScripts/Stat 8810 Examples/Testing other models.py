#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walmart data, but a different model (bt, binomial, poisson, hbart, probit, etc.)
1. bt: works fine but isn't very accurate at all; Update: now no mdraws
2. binomial: no mdraws
3. poisson: no mdraws
4. bart: regular (default model)
5. hbart: works fine
6. probit: no mdraws
7. modifiedprobit: no mdraws
8. merck_truncated: no mdraws
"""
import numpy as np
import matplotlib.pyplot as plt
# from sklearn import datasets, svm
# Janky importing from openbt-python repo (you'll have to change this for your own machine):
import sys
sys.path.append("/home/clark/Documents/OpenBT/openbt-python") # os.getcwd() to check
from openbt2 import OPENBT # I made changes to openbt.py & called it openbt2
sys.path.append("/home/clark/Documents/OpenBT/PyScripts/Walmart Example")
from Construct_Walmart_Data import *
from summarize_output import *

# Load in the data (8 x variables, after I edited it):
(x, y, x_pd, y_pd) = get_walmart_data()

# Settings:
# Tree prior
alpha = 0.95 # Default = 0.95
beta = 2 # Default = 2
# MCMC settings: not looking for an accurate fit yet, just testing!
N = 100 # (AKA ndpost); Default = 1000
burn = 100 # (AKA nskip); Default = 100
nadapt = 100 # default = 1000
adaptevery = 50 # Default = 100
ntreeh = 1 # Default = 1
tc = 5 # Default = 2, but we will use 5 or 6 from now on
shat = np.std(y, ddof = 1)
m=1
k=1
nu=1
nc=100
npred_arr = 40000 # Rows of the preds x grid
preds_list = []; np.random.seed(88)

# Categorical variables:
for col in [0, 1, 2, 3]:
     preds_list.append(np.random.randint(np.min(x[:, col]), np.max(x[:, col])+1, size = npred_arr))
# Continuous variables:
for col in [4, 5, 6, 7]:
     preds_list.append(np.random.uniform(np.min(x[:, col]), np.max(x[:, col])+2.2e-16, size = npred_arr))
preds_test = np.asarray(preds_list).T # This is supposedly faster than having it be a np array the whole time
# print(preds.nbytes / 1000000) # To view storage size in MB

m_test = OPENBT(model="bt", ndpost=N, nadapt = nadapt, nskip=burn, power=beta,
             base=alpha, tc=tc, numcut=nc, ntree=m, ntreeh=ntreeh, k=k,
             overallsd=shat, overallnu=nu)
fit_test = m_test.fit(x,y)
# Now, print the folder where all the config stuff is, so I can inspect:
print(m_test.fpath)
fitp_test = m.predict(preds_test)
# summarize_fitp(fitp_test)


# Next step: See how good the fit is for this "other model":     
"""
# For plotting (not accurate yet):
npreds = 100 # Default = 100
fig = plt.figure(figsize=(10,5.5))
plt.rcParams['axes.labelsize'] = 18; plt.rcParams['axes.titlesize'] = 22;
plt.rcParams['xtick.labelsize'] = 16; plt.rcParams['ytick.labelsize'] = 16;
path = 'Documents/OpenBT/PyScripts/Plots/' # Will be different for your filesystem
fname = 'Testing_Models.png'

# Plot the fitp:
ax = fig.add_subplot(111)
ax.plot(design, y, 'ro') # label=current_time can help with plot versions
ax.set_title(f'Predicted mean response +/- 2 s.d., ntree = {m.ntree}')
# ^ Technically the +/- will be 1.96 SD
ax.set_xlabel('Observed'); ax.set_ylabel('Fitted'); ax.set_ylim(-1.5, 1)
# plt.legend(loc='upper right')
# Plot the central line: Overall mean predictor
ax.plot(preds, m.mmean, 'b-', linewidth=2)

# Make the full plot with the gray lines: (mmeans and smeans are now returned by m.predict()!)
ax.plot(preds, m.mmean - 1.96 * m.smean, color='black', linewidth=0.8)
ax.plot(preds, m.mmean + 1.96 * m.smean, color='black', linewidth=0.8)
if (N < npreds):
     print('Number of posterior draws (ndpost) are less than the number of', 
                'x-values to predict. This is not recommended.')
     npreds = N
for i in range(npreds):
     ax.plot(preds, m.mdraws[i, :], color="gray", linewidth=1, alpha = 0.20)
     plt.savefig(f'{path}{fname}')
"""

# Vartivity:
fitv_test = m_test.vartivity()
# summarize_fitv(fitv)

# Sobol:
fits_test = m_test.sobol(cmdopt = 'MPI', tc = tc)
# summarize_fits(fits)