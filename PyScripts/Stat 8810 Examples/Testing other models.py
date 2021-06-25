#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Slides 11 data, but a different model (bt, binomial, poisson, hbart, probit, etc.)
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
from summarize_output import *
sys.path.append("/home/clark/Documents/OpenBT/PyScripts/Stat 8810 Examples/")
from gen_data11 import *
# Example (Our usual GP realization) originally using BayesTree, 
# now written in Python with openbt-python.
design, y = gen_data()


# Now, set up the fit:
# Set values to be used in the fitting/predicting:
# Variance and mean priors
shat = 0.1 # (AKA shat) # in reality , it's lambdatrue^2? Lower shat --> more fitting, I think
nu = 3; k = 2 # lower k --> more fitting, I think

# Tree prior
alpha = 0.95 # Default = 0.95
beta = 2 # Default = 2
nc = 1000 # (AKA numcut); Default = 100, but we usually use 1000

# MCMC settings
N = 1000 # (AKA ndpost); Default = 1000
burn = 1000 # (AKA nskip); Default = 100
nadapt = 1000 # default = 1000
tc = 6 # Default = 2, but we usually use 4

# For plotting:
npreds = 100 # Default = 100
fig = plt.figure(figsize=(10,5.5))
plt.rcParams['axes.labelsize'] = 18; plt.rcParams['axes.titlesize'] = 22;
plt.rcParams['xtick.labelsize'] = 16; plt.rcParams['ytick.labelsize'] = 16;
path = 'Documents/OpenBT/PyScripts/Plots/' # Will be different for your filesystem
fname = 'Testing_Models.png'

# No fit_pipeline for now, so that I can debug better
m = OPENBT(model="bt", ndpost=N, nskip=burn, power=beta, base=alpha,
                tc=tc, numcut=nc) # Will use model defaults for trees, k, nu, etc.
fit = m.fit(design,y)
preds = np.arange(0, (1 + 1/npreds), 1/(npreds-1)).reshape(npreds, 1)
fitp = m.predict(preds) # This is usually where the "other"" models fail
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

# summarize_fitp(fitp)
# See if vartivity works:
fitv = m.vartivity()
summarize_fitv(fitv)

# Sobol not needed because we have only 1 variable