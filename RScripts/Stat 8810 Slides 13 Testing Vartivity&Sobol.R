library(Ropenbt)

# Example - the CO2 Plume data from Assignment 3
library(rgl)
load("co2plume.dat_")
options(rgl.printRglwidget = T)
plot3d(co2plume)
# ^ Needed this line for the plot to show, btw: options(rgl.printRglwidget = T)
# Info: str(co2plume)

# Fit the model
y=co2plume$co2
x=co2plume[,1:2]
npred_arr = 25
preds=as.data.frame(expand.grid(seq(0,1,length=npred_arr),
                                seq(0,1,length=npred_arr)))
colnames(preds)=colnames(x)
# For testing, we have small numbers for the parameters:
overallsd = 0.2; overallnu = 3; k = 2
alpha = 0.95; beta = 2; nc = 30
N = 700; burn = 30; nadapt = 300; tc = 4; m = 1

# Do this example (CO2 Plume) manually, since it's a different setup than what I wrote the
# function for:
fit13 <- openbt(x, y, tc=tc, pbd=c(0.7,0.0), overallsd=overallsd, overallnu=overallnu,
                ntreeh=1, ntree=m, k=k, model="bart", power=beta, base=alpha,
                numcut=nc, ndpost=N, nskip=burn)
str(fit13)

# Calculate predictions:
fitp13 <- predict.openbt(fit11, x.test=preds, tc=4)
str(fitp13)

fitv13 <- vartivity.openbt(fit13)
fits13 <- sobol.openbt(fit13, tc=4)

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