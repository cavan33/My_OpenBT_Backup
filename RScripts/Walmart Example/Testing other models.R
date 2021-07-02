source("~/Documents/OpenBT/RScripts/Walmart Example/Construct_Walmart_Data.R")
source("~/Documents/OpenBT/RScripts/Walmart Example/summarize_output.R")
source("~/Documents/OpenBT/RScripts/Walmart Example/walmart_pred_plot.R")
source("~/Documents/OpenBT/openbt/Ropenbt/R/openbt2.R") # Make sure to use my local copy
# of the source code for the test of this function. this line replaces "library(Ropenbt)"
data <- get_walmart_data()
x <- data$x; y <- data$y

# Settings:
# Tree prior
alpha = 0.95 # Default = 0.95
beta = 2 # Default = 2
# MCMC settings: not looking for an accurate fit yet, just testing!
N = 100 # (AKA ndpost); Default = 1000
burn = 100 # (AKA nskip); Default = 100
nadapt = 100 # Default = 1000
adaptevery = 50 # Default = 100
ntreeh = 1 # Default = 1
tc = 5 # Default = 2, but we will use 5 or 6 from now on
shat = sd(y)
m=1
k=1
nu=1
nc=100

npred_arr = 1000; set.seed(88)
preds_test <- data.frame()
# Categorical variables:
for (v in 1:4){
  preds_test <- rbind(preds_test, sample(min(x[, v]):max(x[, v]), npred_arr, replace = TRUE))
}
# Continuous variables:
for (v in 5:8){
  preds_test <- rbind(preds_test, runif(npred_arr, min(x[, v]), max(x[, v])))
}
preds_test <- as.data.frame(t(as.matrix(preds_test))); rownames(preds_test) <- NULL
library(pryr); object_size(preds_test) # Should be only < 1 MB (not 100+!)

# rgy = c(-2, 2); fmean.out = NA
fit_test <- openbt(x, y, model="bart", ndpost=N, nadapt=nadapt, nskip=burn, power=beta,
              base=alpha, tc=tc, numcut=nc, ntree=m, ntreeh=ntreeh, k=k, 
              overallsd=shat, overallnu=nu, pbd=c(0.7,0.0))

# bt (1) needed me to specify rgy and fmean.out outside of the function, hmmm
# Also bt hangs up on the influence part of the code. These errors MAY have been
# because it doesn't let you type a string to set the first 3 models

# Diagnostics to find out why the different models aren't working in Python:
print(fit_test$folder)
# Plan: Go to the folder and view the config (which will be not-removed b/c of my source code edits)

fitp_test <- predict.openbt(fit_test, x.test=preds_test, tc=tc)

# Plot y vs yhat plots:
source("~/Documents/OpenBT/RScripts/Walmart Example/walmart_pred_plot.R")
c(ys, yhats) <- set_up_plot(fitp_test, x, y, points = 2000, var = c(1, 2, 3, 4), day_range = 30)



# Vartivity:
fitv_test = vartivity.openbt(fit_test)
# summarize_fitv(fitv)

# Sobol:
fits_test = sobol.openbt(fit_test, tc = tc)
# summarize_fits(fits)