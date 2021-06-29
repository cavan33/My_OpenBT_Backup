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
source("dace.sim.r")

# Generate response:
set.seed(88)
n=5; k=1; rhotrue=0.2; lambdatrue=1
design=as.matrix(runif(n))
l1=list(m1=outer(design[,1],design[,1],"-"))
l.dez=list(l1=l1)
R=rhogeodacecormat(l.dez,c(rhotrue))$R
L=t(chol(R))
u=rnorm(nrow(R))
z=L%*%u

# Our observed data:
y=as.vector(z)



# Load up BayesTree
library(BayesTree)
preds=matrix(seq(0,1,length=100),ncol=1)

# Variance prior
shat=sd(y)
nu=3
q=0.90
# Mean prior
k=2
# Tree prior
m=1
alpha=0.95
beta=2
nc=100
# MCMC settings
N=1000
burn=1000



# Fit BART
fit=bart(design,y,preds,sigest=shat,sigdf=nu,sigquant=q,
         k=k,power=beta,base=alpha,ntree=m,numcut=nc,
         ndpost=N,nskip=burn)
str(fit)



# Plot fitted response
plot(design,y,pch=20,col="red",cex=2,xlim=c(0,1),
     ylim=c(2.3,3.7),xlab="x",
     main="Predicted mean response +/- 2s.d.")
for(i in 1:nrow(fit$yhat.test))
  lines(preds,fit$yhat.test[i,],col="grey",lwd=0.25)
mean=apply(fit$yhat.test,2,mean)
sd=apply(fit$yhat.test,2,sd)
lines(preds,mean-1.96*sd,lwd=0.75,col="black")
lines(preds,mean+1.96*sd,lwd=0.75,col="black")
lines(preds,mean,lwd=2,col="blue")
points(design,y,pch=20,col="red")





# Try m=10 trees
m=10
fit=bart(design,y,preds,sigest=shat,sigdf=nu,sigquant=q,
         k=k,power=beta,base=alpha,ntree=m,numcut=nc,
         ndpost=N,nskip=burn)
str(fit)



# Plot
plot(design,y,pch=20,col="red",cex=2,xlim=c(0,1),
     ylim=c(2.3,3.7),xlab="x",
     main="Predicted mean response +/- 2s.d.")
for(i in 1:nrow(fit$yhat.test))
  lines(preds,fit$yhat.test[i,],col="grey",lwd=0.25)
mean=apply(fit$yhat.test,2,mean)
sd=apply(fit$yhat.test,2,sd)
lines(preds,mean-1.96*sd,lwd=0.75,col="black")
lines(preds,mean+1.96*sd,lwd=0.75,col="black")
lines(preds,mean,lwd=2,col="blue")
points(design,y,pch=20,col="red")




# Try m=20 trees
m=20
fit=bart(design,y,preds,sigest=shat,sigdf=nu,sigquant=q,
         k=k,power=beta,base=alpha,ntree=m,numcut=nc,
         ndpost=N,nskip=burn)
str(fit)



# Plot
plot(design,y,pch=20,col="red",cex=2,xlim=c(0,1),
     ylim=c(2.3,3.7),xlab="x",
     main="Predicted mean response +/- 2s.d.")
for(i in 1:nrow(fit$yhat.test))
  lines(preds,fit$yhat.test[i,],col="grey",lwd=0.25)
mean=apply(fit$yhat.test,2,mean)
sd=apply(fit$yhat.test,2,sd)
lines(preds,mean-1.96*sd,lwd=0.75,col="black")
lines(preds,mean+1.96*sd,lwd=0.75,col="black")
lines(preds,mean,lwd=2,col="blue")
points(design,y,pch=20,col="red")




# Try m=100 trees
m=100
fit=bart(design,y,preds,sigest=shat,sigdf=nu,sigquant=q,
         k=k,power=beta,base=alpha,ntree=m,numcut=nc,
         ndpost=N,nskip=burn)
str(fit)




# Plot
plot(design,y,pch=20,col="red",cex=2,xlim=c(0,1),
     ylim=c(2.3,3.7),xlab="x",
     main="Predicted mean response +/- 2s.d.")
for(i in 1:nrow(fit$yhat.test))
  lines(preds,fit$yhat.test[i,],col="grey",lwd=0.25)
mean=apply(fit$yhat.test,2,mean)
sd=apply(fit$yhat.test,2,sd)
lines(preds,mean-1.96*sd,lwd=0.75,col="black")
lines(preds,mean+1.96*sd,lwd=0.75,col="black")
lines(preds,mean,lwd=2,col="blue")
points(design,y,pch=20,col="red")





# Try m=200 trees, the recommended default
m=200
fit=bart(design,y,preds,sigest=shat,sigdf=nu,sigquant=q,
         k=k,power=beta,base=alpha,ntree=m,numcut=nc,
         ndpost=N,nskip=burn)
str(fit)



# Plot
plot(design,y,pch=20,col="red",cex=2,xlim=c(0,1),
     ylim=c(2.3,3.7),xlab="x",
     main="Predicted mean response +/- 2s.d.")
for(i in 1:nrow(fit$yhat.test))
  lines(preds,fit$yhat.test[i,],col="grey",lwd=0.25)
mean=apply(fit$yhat.test,2,mean)
sd=apply(fit$yhat.test,2,sd)
lines(preds,mean-1.96*sd,lwd=0.75,col="black")
lines(preds,mean+1.96*sd,lwd=0.75,col="black")
lines(preds,mean,lwd=2,col="blue")
points(design,y,pch=20,col="red")




# Try m=200 trees, the recommended default
m=200
# And k=1
k=1
fit=bart(design,y,preds,sigest=shat,sigdf=nu,sigquant=q,
         k=k,power=beta,base=alpha,ntree=m,numcut=nc,
         ndpost=N,nskip=burn)
str(fit)



# Plot
plot(design,y,pch=20,col="red",cex=2,xlim=c(0,1),
     ylim=c(2.3,3.7),xlab="x",
     main="Predicted mean response +/- 2s.d.")
for(i in 1:nrow(fit$yhat.test))
  lines(preds,fit$yhat.test[i,],col="grey",lwd=0.25)
mean=apply(fit$yhat.test,2,mean)
sd=apply(fit$yhat.test,2,sd)
lines(preds,mean-1.96*sd,lwd=0.75,col="black")
lines(preds,mean+1.96*sd,lwd=0.75,col="black")
lines(preds,mean,lwd=2,col="blue")
points(design,y,pch=20,col="red")




# Example
# Try m=200 trees, the recommended default
m=200
# And k=1
k=1
# And nu=3, q=.99
nu=3
q=0.99

fit=bart(design,y,preds,sigest=shat,sigdf=nu,sigquant=q,
         k=k,power=beta,base=alpha,ntree=m,numcut=nc,
         ndpost=N,nskip=burn)
str(fit)




# Plot
plot(design,y,pch=20,col="red",cex=2,xlim=c(0,1),
     ylim=c(2.3,3.7),xlab="x",
     main="Predicted mean response +/- 2s.d.")
for(i in 1:nrow(fit$yhat.test))
  lines(preds,fit$yhat.test[i,],col="grey",lwd=0.25)
mean=apply(fit$yhat.test,2,mean)
sd=apply(fit$yhat.test,2,sd)
lines(preds,mean-1.96*sd,lwd=0.75,col="black")
lines(preds,mean+1.96*sd,lwd=0.75,col="black")
lines(preds,mean,lwd=2,col="blue")
points(design,y,pch=20,col="red")




# Example
# Try m=200 trees, the recommended default
m=200
# And k=1
k=1
# And nu=2, q=.99
nu=2
q=0.99

fit=bart(design,y,preds,sigest=shat,sigdf=nu,sigquant=q,
         k=k,power=beta,base=alpha,ntree=m,numcut=nc,
         ndpost=N,nskip=burn)
str(fit)




# Plot
plot(design,y,pch=20,col="red",cex=2,xlim=c(0,1),
     ylim=c(2.3,3.7),xlab="x",
     main="Predicted mean response +/- 2s.d.")
for(i in 1:nrow(fit$yhat.test))
  lines(preds,fit$yhat.test[i,],col="grey",lwd=0.25)
mean=apply(fit$yhat.test,2,mean)
sd=apply(fit$yhat.test,2,sd)
lines(preds,mean-1.96*sd,lwd=0.75,col="black")
lines(preds,mean+1.96*sd,lwd=0.75,col="black")
lines(preds,mean,lwd=2,col="blue")
points(design,y,pch=20,col="red")




# Example
# Try m=200 trees, the recommended default
m=200
# And k=1
k=1
# And nu=1, q=.99
nu=1
q=0.99

fit=bart(design,y,preds,sigest=shat,sigdf=nu,sigquant=q,
         k=k,power=beta,base=alpha,ntree=m,numcut=nc,
         ndpost=N,nskip=burn)
str(fit)



# Plot
plot(design,y,pch=20,col="red",cex=2,xlim=c(0,1),
     ylim=c(2.3,3.7),xlab="x",
     main="Predicted mean response +/- 2s.d.")
for(i in 1:nrow(fit$yhat.test))
  lines(preds,fit$yhat.test[i,],col="grey",lwd=0.25)
mean=apply(fit$yhat.test,2,mean)
sd=apply(fit$yhat.test,2,sd)
lines(preds,mean-1.96*sd,lwd=0.75,col="black")
lines(preds,mean+1.96*sd,lwd=0.75,col="black")
lines(preds,mean,lwd=2,col="blue")
points(design,y,pch=20,col="red")




# Example
# Try m=200 trees, the recommended default
m=200
# And k=1
k=1
# And nu=1, q=.99
nu=1
q=0.99
# And numcuts=1000
nc=1000

fit=bart(design,y,preds,sigest=shat,sigdf=nu,sigquant=q,
         k=k,power=beta,base=alpha,ntree=m,numcut=nc,
         ndpost=N,nskip=burn)
str(fit)




# Plot
plot(design,y,pch=20,col="red",cex=2,xlim=c(0,1),
     ylim=c(2.3,3.7),xlab="x",
     main="Predicted mean response +/- 2s.d.")
for(i in 1:nrow(fit$yhat.test))
  lines(preds,fit$yhat.test[i,],col="grey",lwd=0.25)
mean=apply(fit$yhat.test,2,mean)
sd=apply(fit$yhat.test,2,sd)
lines(preds,mean-1.96*sd,lwd=0.75,col="black")
lines(preds,mean+1.96*sd,lwd=0.75,col="black")
lines(preds,mean,lwd=2,col="blue")
points(design,y,pch=20,col="red")




# Example - the CO2 Plume data from Assignment 3
library(rgl)
load("co2plume.dat")
options(rgl.printRglwidget = T)
plot3d(co2plume)




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
q=0.99
# And numcuts=1000
nc=1000

fit=bart(x,y,preds,sigest=shat,sigdf=nu,sigquant=q,
         k=k,power=beta,base=alpha,ntree=m,numcut=nc,
         ndpost=N,nskip=burn)




# Plot posterior samples of sigma
plot(fit$sigma,type='l',xlab="Iteration",
     ylab=expression(sigma))





# Plot fit
ym=fit$yhat.test.mean
ysd=apply(fit$yhat.test,2,sd)
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
