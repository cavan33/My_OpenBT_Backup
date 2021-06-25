source("~/Documents/OpenBT/RScripts/Construct_Walmart_Data.R")
source("~/Documents/OpenBT/RScripts/summarize_output.R")
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
tc = 6 # Default = 2, but we will use 6 from now on
shat = sd(y)
m=200
k=1
nu=1
nc=2000
npred_arr = 4
preds=as.data.frame(expand.grid(seq(min(x[,1]),max(x[,1]),length=npred_arr),
                                seq(min(x[,2]),max(x[,2]),length=npred_arr),
                                seq(min(x[,3]),max(x[,3]),length=npred_arr),
                                seq(min(x[,4]),max(x[,4]),length=2),
                                seq(min(x[,5]),max(x[,5]),length=npred_arr),
                                seq(min(x[,6]),max(x[,6]),length=npred_arr),
                                seq(min(x[,7]),max(x[,7]),length=npred_arr),
                                seq(min(x[,8]),max(x[,8]),length=npred_arr)))
# The holiday flag variable has fewer than npred_arr possibilities, so its length is lessened
library(pryr); object_size(preds)
library(Ropenbt)
fit <- openbt(x, y, model="bart", ndpost=N, nadapt=nadapt, nskip=burn, power=beta,
           base=alpha, tc=tc, numcut=nc, ntree=m, ntreeh=ntreeh, k=k, 
           overallsd=shat, overallnu=nu, pbd=c(0.7,0.0))
str(fit)

fitp <- predict.openbt(fit, x.test=preds, tc=tc)
str(fitp)
summarize_fitp(fitp)

# Vartivity:
fitv <- vartivity.openbt(fit)
summarize_fitv(fitv)

# Sobol:
fits <- sobol.openbt(fit, tc=6)
summarize_fits(fits)
save_fits(fits, '/home/clark/Documents/OpenBT/RScripts/walmart_fits_result.txt')