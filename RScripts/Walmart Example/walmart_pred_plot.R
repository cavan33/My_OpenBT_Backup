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

     
     
     
pred_plot <- function(y_df, title, fname, ms = 1, millions = TRUE, lims_df = data.frame()){
    # Plots the output from the previous function.
    # Parameters
    # ----------
    # y_df : dataframe
    #     has 'y' and 'yhat': y_train and y_test to plot.
    # title : string
    #     Custom title of the plot
    # fname : string
    #     File location to which to save the plot
    # ms : double, optional
    #     size (markersize) of points. The default is 1.
    # millions : TYPE, optional
    #     If True, divide all y-values by a million. The default is True.
    # lims_df : dataframe, optional
    #     Specifies limits of the plot (if the defaults aren't good)
    # 
    # Returns
    # -------
    # None.
    library(ggplot2)
    if (millions) {
      y_df$y <- y_df$y / 1000000; y_df$yhat <- y_df$yhat / 1000000;
      x_lab <- 'Data (y), Millions of $'; y_lab <- 'Predicted (yhat), Millions of $'
    } else {
        x_lab <- 'Data (y), $'; y_lab <- 'Predicted (yhat), $'
    }
    if(length(lims_df) == 0){
      lims_df <- data.frame("x" = c(min(y_df$y, y_df$yhat), max(y_df$y, y_df$yhat)),
                         "y" = c(min(y_df$y, y_df$yhat), max(y_df$y, y_df$yhat)))
    }        
    p <- ggplot(y_df, aes(y, yhat)) +
            geom_point(color = "red", size = ms) +
            geom_line(lims_df, mapping = aes(x, y)) +
            labs(x = x_lab, y = y_lab, title = title)
    ggsave(fname, p, device = "png")
    return(p)
}