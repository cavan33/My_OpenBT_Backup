source("~/Documents/OpenBT/RScripts/Walmart Example/Construct_Walmart_Data.R")
source("~/Documents/OpenBT/RScripts/Walmart Example/summarize_output.R")
source("~/Documents/OpenBT/RScripts/Walmart Example/walmart_pred_plot.R")
data <- get_walmart_data()
x <- data$x; y <- data$y

# Settings:
alpha = 0.95 # Default = 0.95
beta = 2 # Default = 2
# MCMC settings
N = 2000 # (AKA ndpost); Default = 1000
burn = 2000 # (AKA nskip); Default = 100
nadapt = 2000 # Default = 1000
adaptevery = 100 # Default = 100
ntreeh = 1 # Default = 1
tc = 5 # Default = 2, but we will use 5 or 6 from now on
shat = sd(y)
m=200
k=1
nu=1
nc=2000

npred_arr = 25000; set.seed(88) # 40000 was overloading my computer!
preds <- data.frame()
# Categorical variables:
for (v in 1:3){
  preds <- rbind(preds, sample(min(x[, v]):max(x[, v]), npred_arr, replace = TRUE))
}
# Special weighted variable (#4): Holiday flag (much more 0s than 1s)
preds <- rbind(preds, sample(0:1, npred_arr, prob = c(1-mean(x[,4]), mean(x[,4])), replace = TRUE))

# Continuous variables:
for (v in 5:8){
  preds <- rbind(preds, runif(npred_arr, min(x[, v]), max(x[, v])))
}
preds <- as.data.frame(t(as.matrix(preds))); rownames(preds) <- NULL
library(pryr); object_size(preds) # Should be only a few MB (not 100+!)

library(Ropenbt)
fit <- openbt(x, y, model="bart", ndpost=N, nadapt=nadapt, nskip=burn, power=beta,
              base=alpha, tc=tc, numcut=nc, ntree=m, ntreeh=ntreeh, k=k, 
              overallsd=shat, overallnu=nu, pbd=c(0.7,0.0))
str(fit)

fitp <- predict.openbt(fit, x.test=preds, tc=tc)
str(fitp)
# summarize_fitp(fitp)

# Vartivity:
fitv <- vartivity.openbt(fit)
# summarize_fitv(fitv)

# Sobol:
fits <- sobol.openbt(fit, tc=tc)
# summarize_fits(fits)
# save_fits(fits, '/home/clark/Documents/OpenBT/RScripts/Walmart Example/Results/fits_result.txt')

# Save objects:
fpath = '/home/clark/Documents/OpenBT/RScripts/Walmart Example/Results/'
save_fit_obj(fit, paste(fpath, 'fit_result', sep=""), 'fit')
# save_fit_obj(fitp, paste(fpath, 'fitp_result', sep=""), 'fitp') # Takes up A LOT of memory
save_fit_obj(fitv, paste(fpath, 'fitv_result', sep=""), 'fitv')
source("~/Documents/OpenBT/RScripts/Walmart Example/summarize_output.R")
save_fit_obj(fits, paste(fpath, 'fits_result', sep=""), 'fits')

# Plot y vs yhat plots:
c(ys, yhats) <- set_up_plot(fitp, x, y, points = length(x), var = c(0, 1, 2, 3), day_range = 30)








