library(Ropenbt)

# Example - the CO2 Plume data from Assignment 3
library(rgl)
load("co2plume.dat_")
options(rgl.printRglwidget = T)
plot3d(co2plume)
# ^ Needed this line for the plot to show, btw: options(rgl.printRglwidget = T)
# Info: str(co2plume)

summarize_fitp <- function(fitp){
  print(head(fitp$mdraws[1,])); print(mean(fitp$mdraws))
  print(head(fitp$sdraws[1,])); print(mean(fitp$sdraws))
  # print(head(fitp$mmean)); print(mean(fitp$mmean)) # Will be the same as the mean values I took down above
  # print(head(fitp$smean)); print(mean(fitp$smean))
  print(head(fitp$msd)); print(mean(fitp$msd))
  print(head(fitp$ssd)); print(mean(fitp$ssd))
  print(mean(fitp$m.5)); print(mean(fitp$m.lower)); print(mean(fitp$m.upper))
  print(mean(fitp$s.5)); print(mean(fitp$s.lower)); print(mean(fitp$s.upper))
}
summarize_fitv <- function(fitv){
  print(fitv$vdraws[1:60, ]); print(mean(fitv$vdraws))
  print(fitv$vdrawsh[1:60, ]); print(mean(fitv$vdrawsh))
  print(fitv$mvdraws); print(fitv$mvdrawsh)
  print(fitv$vdraws.sd); print(fitv$vdrawsh.sd); print(fitv$vdraws.5)
  print(fitv$vdrawsh.5); print(fitv$vdraws.lower); print(fitv$vdraws.upper)
  print(fitv$vdrawsh.lower); print(fitv$vdrawsh.upper)
}
summarize_fits <- function(fits){
  print(mean(fits$vidraws)); print(mean(fits$vijdraws))
  print(mean(fits$tvidraws)); print(mean(fits$vdraws))
  print(mean(fits$sidraws)); print(mean(fits$sijdraws))
  print(mean(fits$tsidraws))
  print(fits$msi); print(fits$msi.sd); print(fits$si.5)
  print(fits$si.lower); print(fits$si.upper); print(fits$msij)
  print(fits$sij.sd); print(fits$sij.5); print(fits$sij.lower)
  print(fits$sij.upper); print(fits$mtsi); print(fits$tsi.sd)
  print(fits$tsi.5); print(fits$tsi.lower); print(fits$tsi.upper)
}

# Settings from further above in the Slides 13 example:
# Tree prior
alpha = 0.95 # Default = 0.95
beta = 2 # Default = 2
# MCMC settings
N = 1000 # (AKA ndpost); Default = 1000
burn = 1000 # (AKA nskip); Default = 100, but I've usually been doing 1000
nadapt = 1000 # Default = 1000
adaptevery = 100 # Default = 100
ntreeh = 1 # Default = 1
tc = 4 # Default = 2
npred_arr = 20

# Fit the model
y=co2plume$co2
x=co2plume[,1:2]
preds=as.data.frame(expand.grid(seq(0,1,length=npred_arr),
                                seq(0,1,length=npred_arr)))
colnames(preds)=colnames(x)
shat = sd(y)
# Try m=200 trees, the recommended default
m=200
# And k=1
k=1
# And nu=1, q=.99
nu=1
# And numcuts=1000
nc=1000

# Do this example (CO2 Plume) manually, since it's a different setup than what I wrote the
# function for:
fit13 <- openbt(x, y, tc=tc, pbd=c(0.7,0.0), overallsd=shat, overallnu=nu,
                ntreeh=1, ntree=m, k=k, model="bart", power=beta, base=alpha,
                numcut=nc, ndpost=N, nskip=burn, nadapt=nadapt)
str(fit13)

# Calculate predictions:
fitp13 <- predict.openbt(fit13, x.test=preds, tc=4)
# str(fitp13)
summarize_fitp(fitp13)

fitv13 <- vartivity.openbt(fit13)
summarize_fitv(fitv13)

fits13 <- sobol.openbt(fit13, tc=4)
summarize_fits(fits13)
#----------------------------------------------------------------------------------------
# Testing/Hard-coding section: 
# This block goes with CO2Plume Example - trying to find what exactly config.sobol contains
c(fit13$modelname,fit13$xiroot,
  paste(fit13$nd),paste(fit13$m),
  paste(fit13$mh),paste(length(fit13$xicuts)),paste(fit13$minx),
  paste(fit13$maxx),paste(fit13$tc))



# Trying to manually run the function to see what "draws" is supposed to look like
res=list(); tc = 4
draws=read.table('/tmp/openbtpy_0322h0dp/model.sobol0')
for(i in 2:tc)
  draws=rbind(draws,read.table(paste('/tmp/openbtpy_0322h0dp/model.sobol',i-1,sep="")))
draws=as.matrix(draws)
# Trying to figure out what this means (complete!):
p=4
labs=gsub("\\s+",",",apply(combn(1:p,2),2,function(zz) Reduce(paste,zz)))