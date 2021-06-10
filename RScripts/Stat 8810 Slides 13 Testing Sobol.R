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
preds=as.data.frame(expand.grid(seq(0,1,length=20),
                                seq(0,1,length=20)))
colnames(preds)=colnames(x)
shat=sd(y)
# Try m=200 trees, the recommended default
m=200
# And k=1
k=1
# And nu=1, q=.99
nu=1
# And numcuts=1000
nc=1000
# Do this one manually, since it's a different setup than what I wrote the
# function for:
fit11 <- openbt(x, y, tc=4, pbd=c(0.7,0.0), overallsd=shat, overallnu=nu,
                ntreeh=1, ntree=m, k=k, model="bart", power=beta, base=alpha,
                numcut=nc, ndpost=N, nskip=burn)
str(fit11)

# Calculate predictions:
fitp11 <- predict.openbt(fit11, x.test=preds, tc=4)
str(fitp11)

fits11 <- sobol.openbt(fit11, tc=4)

#----------------------------------------------------------------------------------------
# Testing/Hard-coding section:
# Trying to manually run the function to see what "draws" is supposed to look like
res=list(); tc = 4
draws=read.table('/tmp/openbtpy_0322h0dp/model.sobol0')
for(i in 2:tc)
  draws=rbind(draws,read.table(paste('/tmp/openbtpy_0322h0dp/model.sobol',i-1,sep="")))
draws=as.matrix(draws)
# Trying to figure out what this means:
p=4
labs=gsub("\\s+",",",apply(combn(1:p,2),2,function(zz) Reduce(paste,zz)))