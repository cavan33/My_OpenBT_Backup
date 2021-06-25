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
from sklearn import datasets, svm
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
ntree = 1 # (AKA m); Default = 1
ntreeh = 1 # Default = 1

# For plotting:
npreds = 30 # Default = 100
fig = plt.figure(figsize=(10,5.5))
plt.rcParams['axes.labelsize'] = 18; plt.rcParams['axes.titlesize'] = 22;
plt.rcParams['xtick.labelsize'] = 16; plt.rcParams['ytick.labelsize'] = 16;
path = 'Documents/OpenBT/PyScripts/Plots/' # Will be different for your filesystem
fname = 'Slides11_testing_config.png'
#---------------------------------------------------------------------------------------
def fit_pipeline(design, y, model, ndpost, nskip, power, base, tc, numcut, ntree,
                 ntreeh, k, overallsd, overallnu, npreds, fig, path, fname):
     m = OPENBT(model=model, ndpost=ndpost, nskip=nskip, power=power, base=base,
                tc=tc, numcut=numcut, ntree=ntree, ntreeh=ntreeh, k=k,
                overallsd=overallsd, overallnu=overallnu)
     fit = m.fit(design,y)
     preds = np.arange(0, (1 + 1/npreds), 1/(npreds-1)).reshape(npreds,1)
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
     ax.plot(preds, m.mmeans, 'b-', linewidth=2)

     # Make the full plot with the gray lines: (mmeans and smeans are now returned by m.predict()!)
     ax.plot(preds, m.mmeans - 1.96 * m.smean, color='black', linewidth=0.8)
     ax.plot(preds, m.mmeans + 1.96 * m.smean, color='black', linewidth=0.8)
     for i in range(npreds):
          ax.plot(preds, m.mpreds[:,i],color="gray", linewidth=1, alpha = 0.20)
     plt.savefig(f'{path}{fname}')
     return((fig, m)) # ^ Returns the plot and the instance of the class

#---------------------------------------------------------------------------------------
# Fit BART
(plot, m) = fit_pipeline(design, y, model="bart", ndpost=N, nskip=burn,
               power=beta, base=alpha, tc=tc, numcut=nc, ntree=ntree,
               ntreeh=ntreeh, k=k, overallsd=overallsd, overallnu=overallnu,
               npreds=npreds, fig=fig, path=path, fname=fname)
# plt.clf()

fitv = m.vartivity()



"""
# Archive (just in case): Old way to get mmean(s) and smean(s):
     mtemp = np.empty(npreds) # temp array to get mmean; this is also = fitp
     stemp = np.empty(npreds) # temp array to get smean
     for i in range(npreds):
          mtemp[i] = np.mean(m.mpreds[i,:])
          stemp[i] = np.mean(m.spreds[i,:]) # Had to transpose spreds in the openbt file because mpreds and spreds were opposite shapes
          ax.plot(preds, m.mpreds[:,i],color="gray", linewidth=1, alpha = 0.14)
     mmean = np.mean(mtemp) # Not needed, but this is the overall predicted mean of the data
     smean = np.mean(stemp) # Overall predicted SD of the data
     print(mtemp); print(mtemp.shape)
     print(stemp); print(stemp.shape)
     print(mmean); print(smean)
"""