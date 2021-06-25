#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
From this example: http://www.matthewpratola.com/teaching/stat8810-fall-2017/,
Slides 11 Code. It uses the BayesTree package, but it's similar to using OpenBT.

This script replicates the OpenBT fit behavior using Python. I took functions from
Zoltan Puha's repo, but made a new config file called openbt2.py which was tailored
to how I wanted to set some more parameters.
"""
import numpy as np
import matplotlib.pyplot as plt
# from sklearn import datasets, svm
# Janky importing from openbt-python repo (you'll have to change this for your own machine):
import sys
sys.path.append("/home/clark/Documents/OpenBT/openbt-python") # os.getcwd() to check
from openbt2 import OPENBT # I made changes to openbt.py & called it openbt2
sys.path.append("/home/clark/Documents/OpenBT/PyScripts/Stat 8810 Examples/")
from gen_data11 import *
# Example (Our usual GP realization) originally using BayesTree, 
# now written in Python with openbt-python.
design, y = gen_data()


# Now, set up the fit:
# Set values to be used in the fitting/predicting:
# Variance and mean priors
shat = 0.1 # (AKA shat) # in reality , it's lambdatrue^2?
# lower shat --> more fitting, I think
nu = 3
k = 2 # lower k --> more fitting, I think

# Tree prior
alpha = 0.95 # Default = 0.95
beta = 2 # Default = 2
nc = 1000 # (AKA numcut); Default = 100, but we usually use 1000

# MCMC settings
N = 1000 # (AKA ndpost); Default = 1000
burn = 1000 # (AKA nskip); Default = 100
nadapt = 1000 # default = 1000
tc = 4 # Default = 2, but we usually use 4
ntree = 200 # (AKA m); Default = 1, but 200 is commonly used for real fits
ntreeh = 1 # Default = 1

# For plotting:
npreds = 100 # Default = 100
fig = plt.figure(figsize=(10,5.5))
plt.rcParams['axes.labelsize'] = 18; plt.rcParams['axes.titlesize'] = 22;
plt.rcParams['xtick.labelsize'] = 16; plt.rcParams['ytick.labelsize'] = 16;
path = 'Documents/OpenBT/PyScripts/Plots/' # Will be different for your filesystem
fname = 'Slides11_testing.png'
#---------------------------------------------------------------------------------------
def fit_pipeline(design, y, model, ndpost, nskip, power, base, tc, numcut, ntree,
                 ntreeh, k, overallsd, overallnu, npreds, fig, path, fname):
     m = OPENBT(model=model, ndpost=ndpost, nskip=nskip, power=power, base=base,
                tc=tc, numcut=numcut, ntree=ntree, ntreeh=ntreeh, k=k,
                overallsd=overallsd, overallnu=overallnu)
     fit = m.fit(design,y)
     preds = np.arange(0, (1 + 1/npreds), 1/(npreds-1)).reshape(npreds, 1)
     fitp = m.predict(preds)

     # Plot predictions:
     # from datetime import datetime
     # current_time = datetime.now().strftime("%H:%M:%S")
     ax = fig.add_subplot(111)
     ax.plot(design, y, 'ro') # label=current_time can help with plot versions
     ax.set_title(f'Predicted mean response +/- 2 s.d., ntree = {ntree}')
     # ^ Technically the +/- will be 1.96 SD
     ax.set_xlabel('Observed'); ax.set_ylabel('Fitted'); ax.set_ylim(-1.5, 1)
     # plt.legend(loc='upper right')
     # Plot the central line: Overall mean predictor
     ax.plot(preds, m.mmean, 'b-', linewidth=2)

     # Make the full plot with the gray lines: (mmeans and smeans are now returned by m.predict()!)
     ax.plot(preds, m.mmean - 1.96 * m.smean, color='black', linewidth=0.8)
     ax.plot(preds, m.mmean + 1.96 * m.smean, color='black', linewidth=0.8)
     if (ndpost < npreds):
          print('Number of posterior draws (ndpost) are less than the number of', 
                'x-values to predict. This is not recommended.')
          npreds = ndpost
     for i in range(npreds):
          ax.plot(preds, m.mdraws[i, :],color="gray", linewidth=1, alpha = 0.20)
     plt.savefig(f'{path}{fname}')
     return((fig, fit, fitp, m)) # Returns the plot, fit results, and the instance of the class

#---------------------------------------------------------------------------------------
# Fit BART
(plot, fit11, fitp11, m11) = fit_pipeline(design, y, model="bart", ndpost=N, nskip=burn,
               power=beta, base=alpha, tc=tc, numcut=nc, ntree=ntree,
               ntreeh=ntreeh, k=k, overallsd=shat, overallnu=nu,
               npreds=npreds, fig=fig, path=path, fname=fname)
# plt.clf()

fitv11 = m11.vartivity()
# fits (sobol function) # doesn't work for this example because p = 1

# summarize_fitp(fitp11)
# summarize_fitv(fitv11)

"""
thingy = np.arange(8.0)
np.array_split(thingy, 0)
"""