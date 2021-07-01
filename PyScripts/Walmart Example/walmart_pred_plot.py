#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finds the row in the x-test that has the closest variable values to the x-train data,
and pairs their y and y-hat values for plotting.
"""
import numpy as np
import matplotlib.pyplot as plt

def set_up_plot(fitp, x, y, points = 2000, var = [0, 1, 3]):
     """
     Makes two arrays: y, and y-hat respectively, to plot. When plotted, they'll 
     show how close the fit is to predicting the correct y for a (predetermined) 
     set of variables.
     Parameters
     ----------
     fitp : dict
          fit predictions; will give us x_test and y_test arrays.
     x : numpy array
          x_train (data).
     y : numpy_array
          y_train (data responses).
     points : int, optional
          Determines how many points will be plotted - a random sample of
          length (points) of x is taken. The default is 2000.
     var : list, optional
          Specifies which variables to match      

     Returns
     -------
     y1, y2: y and yhat to plot
     """
     x_test = fitp['x_test'] # trials x p (8 for Walmart)
     y_test = fitp['mmean'] # trials x 1
     if(points > len(x)): points = len(x) # Capped by number of actual data points
     np.random.seed(88); count = 0 # Counting the loops where no x_train matches
     idxs = np.sort(np.random.choice(len(x), size = points, replace = False)) # Which points to actually use
     # print(idxs[0:25]) # A check
     y1 = np.empty(points); y2 = np.empty(points)
     for i in range(points): 
          # Find the matching x_train:
          good_idx = {}
          for v in var:
              good_idx[v] = np.where(x_test[:, v] == x[idxs[i], v])[0]
              # Add an if statement here to deal with the day column?
          if (len(var) == 1):
              good_idx_tot = good_idx[var[0]]
          elif (len(var) == 2):
              good_idx_tot = np.array(np.sort(list(set(
              good_idx[var[0]].intersection(good_idx[var[1]])))))
          elif (len(var) == 3):
              good_idx_tot = np.array(np.sort(list(set(set(
              good_idx[var[0]]).intersection(good_idx[var[1]])).intersection(good_idx[var[2]]))))
          elif (len(var) == 4):
              good_idx_tot = np.array(np.sort(list(set(set(set(
              good_idx[var[0]]).intersection(good_idx[var[1]]).intersection(good_idx[var[2]])).intersection(good_idx[var[3]])))))
          else: print("Variable comparisons list error; doesn't have 1-4 variables.")
          if (len(good_idx_tot) > 0):
              y1[i] = y[i] # y_train
              y2[i] = np.round(np.mean(y_test[good_idx_tot]), 2) # mean of y_tests that matched
          else:
              count = count + 1
     # print(good_idx_tot); print(good_idx) # The last one of each; a check
     
     print('Number of x_train rows which were not perfectly matched in x_test:', count)
     # Delete the unfilled rows with 3 steps of masking:
     mask1 = (y2 > 10**(-9))
     y1 = y1[mask1]; y2 = y2[mask1]
     # ^ To get rid of the empties from the count thing
     mask2 = 0 < y2
     y1 = y1[mask2]; y2 = y2[mask2]
     mask3 = y2 < 1e+12
     y1 = y1[mask3]; y2 = y2[mask3]
     # ^ To get rid of a few wacky prediction values
     return(y1, y2) 



def pred_plot(y1, y2, title, fname, ms = 4, millions = True, lims = []):
    """
    Plots the output from the previous function.
    Parameters
    ----------
    y1 : numpy array
        AKA 'y': y_train array to plot.
    y2 : numpy array
        AKA 'y-hat': y_test array to plot.
    title : string
        Custom title of the plot
    fname : string
        File location to which to save the plot
    ms : int, optional
        markersize of points. The default is 4.
    millions : TYPE, optional
        If True, divide all y-values by a million. The default is True.
    lims : list, optional
        Specifies limits of the plot (if the defaults aren't good)   

    Returns
    -------
    None.

    """
    plt.rcParams['axes.labelsize'] = 18; plt.rcParams['axes.titlesize'] = 22;
    plt.rcParams['xtick.labelsize'] = 16; plt.rcParams['ytick.labelsize'] = 16;
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    if millions:
        ax.plot(y1/1000000, y2/1000000, 'ro', markersize = ms)
        ax.set_xlabel('Data (y), Millions of $'); ax.set_ylabel('Predicted (yhat), Millions of $')
    else:
        ax.plot(y1, y2, 'ro', markersize = ms)
        ax.set_xlabel('Data (y), Millions of $'); ax.set_ylabel('Predicted (yhat), Millions of $')
    if(lims == []):
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()])]  # max of both axes
    ax.plot(lims, lims, 'k-', linewidth=2); ax.set_xlim(lims); ax.set_ylim(lims);
    ax.set_title(title)
    plt.savefig(fname)