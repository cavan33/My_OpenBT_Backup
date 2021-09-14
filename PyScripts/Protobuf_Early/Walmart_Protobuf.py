#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Import the Walmart data and run sobol/fitv on it to observe and plot their differences.
"""
import numpy as np; import random
import matplotlib.pyplot as plt
import sys
# This is going to be run in the terminal, so I have to import the OPENBT class a bit differently:
sys.path.append("/home/clark/Documents/OpenBT/openbt-python") # Might be different for your filesystem
print(sys.path)
from openbt2 import OPENBT
sys.path.append("/home/clark/Documents/OpenBT/PyScripts/Walmart Example/Functions")
from Construct_Walmart_Data import *
from summarize_output import *
from walmart_pred_plot import *

# Load in the data (8 x variables, after I edited it):
(x, y, x_pd, y_pd) = get_walmart_data()

# Settings: Low, since we're just testing a new way to save fits
alpha = 0.95 # Default = 0.95
beta = 2 # Default = 2
# MCMC settings
N = 400 # (AKA ndpost); Default = 1000
burn = 40 # (AKA nskip); Default = 100
nadapt = 300 # Default = 1000
adaptevery = 100 # Default = 100
ntreeh = 1 # Default = 1
tc = 5 # Default = 2
shat = np.std(y, ddof = 1)
m=10
k=1
nu=1
nc=30

# Sidebar: For non-in-sample predictions: See orig. Walmart Example file

preds = x # For in-sample
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
fits = m.sobol(cmdopt = 'MPI', tc = tc)
# summarize_fits(fits)

#---------------------------------------------------------------------------------------------
# Save "fit" to protobuf serialized file:
# print(fit) # Just to remember all its attributes/methods
import OpenBT_pb2 as OpenBT_proto # Only works if the file is run in the terminal, sadly.
fit_proto = OpenBT_proto.fit()
fit_proto.ndpost.append(fit['ndpost']) # Direct assignment with "=" isn't allowed, dang
# fit_proto.model = fit['model']
# Many more set statements...
print(fit_proto)

# Now save the fit to a Serialized (binary?) file:
with open("./serializedFile", "wb") as fd:
    fd.write(fit_proto.SerializeToString())

fit_proto = OpenBT_proto.fit()
with open("./serializedFile", "rb") as fd:
    fit_proto.ParseFromString(fd.read())

print(fit_proto) # Shows that we can read in stored binary data and it still works

# The rest is all visualization stuff - likely not relevant for Protobuf testing.
# Go back to Walmart_fit_with... file to get it, though