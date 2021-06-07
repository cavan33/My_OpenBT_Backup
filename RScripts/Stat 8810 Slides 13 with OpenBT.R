#
# Bayesian Additive Regression Tree Model
# For STAT8810
# Fall, 2017
# M.T. Pratola



# BART's prior on the mu's
x=seq(-0.75,0.75,length=1000)
k=2
m=200
f=dnorm(x,mean=0,sd=0.5/(k*sqrt(m)))
f1=dnorm(x,mean=0,sd=0.5/k)
plot(x,f,type='l',lwd=5,xlab=expression(mu[ji]),ylab="Density",main="Prior with m=200, k=2")
lines(x,f1,lty=3,lwd=3,col="pink")
abline(v=-.5,lty=4,col="grey")
abline(v=.5,lty=4,col="grey")



# Example - our usual GP realization
rhogeodacecormat<-function(geoD,rho,alpha=2)
{
  rho=as.vector(rho)
  if(!is.vector(rho)) stop("non-vector rho!")
  if(any(rho<0)) stop("rho<0!")
  if(any(rho>1)) stop("rho>1!")
  if(any(alpha<1) || any(alpha>2)) stop("alpha out of bounds!")
  if(!is.list(geoD)) stop("wrong format for distance matrices list!")
  if(length(geoD)!=length(rho)) stop("rho vector doesn't match distance list")
  
  R=matrix(1,nrow=nrow(geoD$l1$m1),ncol=nrow(geoD$l1$m1))
  for(i in 1:length(rho))
    R=R*rho[i]^(geoD[[i]][[1]]^alpha)
  
  return(list(R=R))
}

# Generate response:
set.seed(88)
n=10; k=1; rhotrue=0.2; lambdatrue=1
design=as.matrix(runif(n))
l1=list(m1=outer(design[,1],design[,1],"-"))
l.dez=list(l1=l1)
R=rhogeodacecormat(l.dez,c(rhotrue))$R
L=t(chol(R))
u=rnorm(nrow(R))
z=L%*%u

# Our observed data:
y=as.vector(z)


# Load up Ropenbt:
library(Ropenbt)
preds=matrix(seq(0,1,length=100),ncol=1)


# Variance prior
shat=0.1 # sd(y) # lower shat leads to more fitting, I think
nu=3
q=0.90
# Mean prior
k=2 # lower k leads to more fitting, I think
# Tree prior
m=1
alpha=0.95
beta=2
nc=100
# MCMC settings
N=1000
burn=1000


openbtbart <- function(design, y, tc, pbd, overallsd, overallnu, ntreeh, ntree,
                       k, model, power, base, numcut, ndpost, nskip, preds,
                       title)
{
  fit=openbt(design, y, tc=4, pbd=c(0.7,0.0), overallsd=shat, overallnu=nu,
             ntreeh=1, ntree=m, k=k, model="bart", power=beta, base=alpha,
             numcut=nc, ndpost=N, nskip=burn)
  str(fit)
  
  # Calculate predictions:
  fitp=predict.openbt(fit, x.test=preds, tc=4)
  str(fitp)
  
  # Plot fitted response:
  plot(design,y,pch=20,col="red",cex=2,xlim=c(0,1),
       ylim=c(-1.5,1),xlab="x",
       main=stringr::str_interp(title)) # plot observed pts
  lines(preds,fitp$mmean,col="blue",xlab="observed",ylab="fitted", lwd=2)
  for(i in 1:length(fitp$mmean))
    lines(preds,fitp$mdraws[i,],col="grey",lwd=0.25)
  lines(preds,fitp$mmean-1.96*fitp$smean,lwd=0.75,col="black")
  lines(preds,fitp$mmean+1.96*fitp$smean,lwd=0.75,col="black")
  lines(preds,fitp$mmean,lwd=2,col="blue")
  points(design,y,pch=20,col="red")
  return(fitp)
}


# Fit BART (with OpenBT package now)
fitp1 <- openbtbart(design, y, tc=4, pbd=c(0.7,0.0), overallsd=shat, overallnu=nu,
           ntreeh=1, ntree=m, k=k, model="bart", power=beta, base=alpha,
           numcut=nc, ndpost=N, nskip=burn, preds=preds, 
           title="Predicted mean response +/- 2s.d., m = ${m}")

# Try m=10 trees
m=10
fitp2 <- openbtbart(design, y, tc=4, pbd=c(0.7,0.0), overallsd=shat, overallnu=nu,
           ntreeh=1, ntree=m, k=k, model="bart", power=beta, base=alpha,
           numcut=nc, ndpost=N, nskip=burn, preds=preds, 
           title="Predicted mean response +/- 2s.d., m = ${m}")



# Try m=20 trees
m=20
fitp3 <- openbtbart(design, y, tc=4, pbd=c(0.7,0.0), overallsd=shat, overallnu=nu,
           ntreeh=1, ntree=m, k=k, model="bart", power=beta, base=alpha,
           numcut=nc, ndpost=N, nskip=burn, preds=preds, 
           title="Predicted mean response +/- 2s.d., m = ${m}")


# Try m=100 trees
m=100
fitp4 <- openbtbart(design, y, tc=4, pbd=c(0.7,0.0), overallsd=shat, overallnu=nu,
           ntreeh=1, ntree=m, k=k, model="bart", power=beta, base=alpha,
           numcut=nc, ndpost=N, nskip=burn, preds=preds, 
           title="Predicted mean response +/- 2s.d., m = ${m}")



# Try m=200 trees, the recommended default
m=200
fitp5 <- openbtbart(design, y, tc=4, pbd=c(0.7,0.0), overallsd=shat, overallnu=nu,
           ntreeh=1, ntree=m, k=k, model="bart", power=beta, base=alpha,
           numcut=nc, ndpost=N, nskip=burn, preds=preds, 
           title="Predicted mean response +/- 2s.d., m = ${m}")



# Try m=200 trees, the recommended default
m=200
# And k=1
k=1
fitp6 <- openbtbart(design, y, tc=4, pbd=c(0.7,0.0), overallsd=shat, overallnu=nu,
           ntreeh=1, ntree=m, k=k, model="bart", power=beta, base=alpha,
           numcut=nc, ndpost=N, nskip=burn, preds=preds, 
           title="Predicted mean response +/- 2s.d., m = ${m}")


#---------------------------------------------------------------------
# For all other runs here, it's OK to ignore q, since openbt currently doesn't
# have a setting for it.
# Example
# Try m=200 trees, the recommended default
m=200
# And k=1
k=1
# And nu=3, q=.99
nu=3
m=200
# And k=1
k=1
fitp7 <- openbtbart(design, y, tc=4, pbd=c(0.7,0.0), overallsd=shat, overallnu=nu,
           ntreeh=1, ntree=m, k=k, model="bart", power=beta, base=alpha,
           numcut=nc, ndpost=N, nskip=burn, preds=preds, 
           title="Predicted mean response +/- 2s.d., m = ${m}")



# Example
# Try m=200 trees, the recommended default
m=200
# And k=1
k=1
# And nu=2, q=.99
nu=2
fitp8 <- openbtbart(design, y, tc=4, pbd=c(0.7,0.0), overallsd=shat, overallnu=nu,
           ntreeh=1, ntree=m, k=k, model="bart", power=beta, base=alpha,
           numcut=nc, ndpost=N, nskip=burn, preds=preds, 
           title="Predicted mean response +/- 2s.d., m = ${m}")


# Example
# Try m=200 trees, the recommended default
m=200
# And k=1
k=1
# And nu=1, q=.99
nu=1
fitp9 <- openbtbart(design, y, tc=4, pbd=c(0.7,0.0), overallsd=shat, overallnu=nu,
           ntreeh=1, ntree=m, k=k, model="bart", power=beta, base=alpha,
           numcut=nc, ndpost=N, nskip=burn, preds=preds, 
           title="Predicted mean response +/- 2s.d., m = ${m}")



# Example
# Try m=200 trees, the recommended default
m=200
# And k=1
k=1
# And nu=1, q=.99
nu=1
# And numcuts=1000
nc=1000
fitp10 <- openbtbart(design, y, tc=4, pbd=c(0.7,0.0), overallsd=shat, overallnu=nu,
           ntreeh=1, ntree=m, k=k, model="bart", power=beta, base=alpha,
           numcut=nc, ndpost=N, nskip=burn, preds=preds, 
           title="Predicted mean response +/- 2s.d., m = ${m}")


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

# Save fit:
openbt.save(fit11,"slides13co2fit")

# Load fitted model to a new object (just for learning/testing purposes):
# fit2=openbt.load("slides11fit")

# Calculate predictions:
fitp11 <- predict.openbt(fit11, x.test=preds, tc=4)
str(fitp11)


#--------------------------------------------------------------------

# Plot posterior samples of sigma
plot(fitp11$sdraws[,1],type='l',xlab="Iteration",
     ylab=expression(sigma), main = "sdraws during the fitp")
# ^ This is a bit weird, but it seems to have the right jaggedy shape.


# Plot fit
ym=fitp11$mmean
ysd=fitp11$smean
persp3d(x=seq(0,1,length=20),y=seq(0,1,length=20),z=matrix(ym,20,20),
        col="grey",xlab="stack_inerts",ylab="time",zlab="CO2")
plot3d(co2plume,add=TRUE)




# Plot fit and uncertainties
persp3d(x=seq(0,1,length=20),y=seq(0,1,length=20),z=matrix(ym,20,20),
        col="grey",xlab="stack_inerts",ylab="time",zlab="CO2")
persp3d(x=seq(0,1,length=20),y=seq(0,1,length=20),
        z=matrix(ym+2*ysd,20,20),col="green",add=TRUE)
persp3d(x=seq(0,1,length=20),y=seq(0,1,length=20),
        z=matrix(ym-2*ysd,20,20),col="green",add=TRUE)
plot3d(co2plume,add=TRUE)