#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Finds the row in the x-test that has the closest variable values to the x-train data,
# pairs their y and y-hat values for plotting, and plots.

set_up_plot <- function(fitp, x, y, points = 2000, var = c(0, 1, 2, 3), day_range = 30){
     # Makes two arrays: y, and y-hat respectively, to plot. When plotted, they'll 
     # show how close the fit is to predicting the correct y for a (predetermined) 
     # set of variables.
     # Parameters
     # ----------
     # fitp : list
     #      fit predictions; will give us x_test and y_test arrays.
     # x : double?
     #      x_train (data).
     # y : double?
     #      y_train (data responses).
     # points : integer, optional
     #      Determines how many points will be plotted - a random sample of
     #      length (points) of x is taken. The default is 2000.
     # var : list, optional
     #      Specifies which variables to match   
     # day_range : integer, optional
     # Allowance for the days variable (~1-1000) to not exactly match 
     # the x_train day number in question. i.e. any entry within (day_offset) 
     # days will be counted as a match. 25-40 seems to be a good value for this. 
     # The default is 30.
     # 
     # Returns
     # -------
     # y1, y2: y and yhat to plot

     x_test <- fitp$x.test # trials x p (8 for Walmart)
     y_test <- fitp$mmean # trials x 1
     if(points > length(x[,1])){points = length(x[,1])} # Capped by number of actual data points
     set.seed(88); count = 0 # Counting the loops where no x_train matches
     idxs = sort(sample(length(x[,1]), size = points, replace = FALSE)) # Which points to actually use
     # print(idxs[1:25]) # A check
     y1 = numeric(0); y2 = numeric(0)
     for (i in 1:points) { 
        # Find the matching x_train:
        good_idx_temp <- vector(mode = "list", length = length(var)); names(good_idx_temp) <- var
        for (v in var) {
          if (v != 3) { # Anything but the day variable:
             good_idx_temp$v <- ifelse(x_test[, v] == x[idxs[i], v], 1, NA) }
          else { # Day variable
             good_idx_temp$v <- ifelse(abs(x_test[, v] - x[idxs[i], v]) <= day_range, 1, NA) }
          for (idx in 1:length(good_idx_temp$v)) {
             good_idx$v[idx] <- good_idx_temp$v[idx]*idx
          }
          good_idx$v <- good_idx$v[!is.na(good_idx$v)]
          print(v); print(good_idx$v)
        }
        
        if (length(var) == 1) {
           good_idx_tot <-  good_idx[var[1]] }
        else { if (length(var) == 2) {
           good_idx_tot <- intersect(good_idx[var[1]], good_idx[var[2]])  }
        else { if (length(var) == 3) {
            good_idx_tot <- intersect(good_idx[var[1]], 
                              intersect(good_idx[var[2]], good_idx[var[3]])) }
        else { if (length(var) == 4) {
            good_idx_tot <- intersect(intersect(good_idx[var[1]], good_idx[var[2]]),
                              intersect(good_idx[var[2]], good_idx[var[3]])) }
        else { print("Variable comparisons list error; doesn't have 1-4 variables.") }  
        }  
        }
        }
        print(good_idx_tot)
        if (length(good_idx_tot) > 0) {
           y1[i] <- y[i] # y_train
           y2[i] <- round(mean(y_test[as.numeric(good_idx_tot)]), 2) # mean of y_tests that matched
        }
        else {
           count <- count + 1 }
     }
     print(good_idx_tot); print(length(good_idx[3])) # A check (on the last iteration of the loop)
     print(paste("Number of x_train rows which were not perfectly matched in x_test: ", count, sep=""))
     # Delete the unfilled rows with masking:
     mask1 <- 0 < y2
     y1 <-  y1[mask1]; y2 <- y2[mask1]
     mask2 <- y2 < 1e+12
     y1 <- y1[mask2]; y2 <- y2[mask2]
     # ^ To get rid of a few wacky prediction values
     return (c(y1, y2))
}    

     
     
     
# pred_plot <-  function(y1, y2, title, fname, ms = 4, millions = True, lims = []){
#        """
#     Plots the output from the previous function.
#     Parameters
#     ----------
#     y1 : numpy array
#         AKA 'y': y_train array to plot.
#     y2 : numpy array
#         AKA 'y-hat': y_test array to plot.
#     title : string
#         Custom title of the plot
#     fname : string
#         File location to which to save the plot
#     ms : int, optional
#         markersize of points. The default is 4.
#     millions : TYPE, optional
#         If True, divide all y-values by a million. The default is True.
#     lims : list, optional
#         Specifies limits of the plot (if the defaults aren't good)   
# 
#     Returns
#     -------
#     None.
# 
#     """
#      plt.rcParams['axes.labelsize'] = 18; plt.rcParams['axes.titlesize'] = 22;
#      plt.rcParams['xtick.labelsize'] = 16; plt.rcParams['ytick.labelsize'] = 16;
#      fig = plt.figure(figsize=(16,9))
#      ax = fig.add_subplot(111)
#      if millions:
#        ax.plot(y1/1000000, y2/1000000, 'ro', markersize = ms)
#      ax.set_xlabel('Data (y), Millions of $'); ax.set_ylabel('Predicted (yhat), Millions of $')
#      else:
#        ax.plot(y1, y2, 'ro', markersize = ms)
#      ax.set_xlabel('Data (y), Millions of $'); ax.set_ylabel('Predicted (yhat), Millions of $')
#      if(lims == []):
#        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
#                np.max([ax.get_xlim(), ax.get_ylim()])]  # max of both axes
#      ax.plot(lims, lims, 'k-', linewidth=2); ax.set_xlim(lims); ax.set_ylim(lims);
#      ax.set_title(title)
#      plt.savefig(fname)
# }