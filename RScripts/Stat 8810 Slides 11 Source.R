#
# Single Tree Models
# For STAT8810
# Fall, 2017
# M.T. Pratola



# Example: Unconditional Realization
set.seed(88)
cuts=seq(0.1,0.9,length=9)
nonterms=c()
terms=c()
stop=FALSE
alpha=0.95
beta=2

# Node 1
d=0
psplit=alpha*(1+d)^(-beta)
runif(1)<psplit
nonterms=c(1)

# Nodes 2,3
d=1
# Node 2
psplit=alpha*(1+d)^(-beta)
runif(1)<psplit
nonterms=c(nonterms,2)
# Node 3
psplit=alpha*(1+d)^(-beta)
runif(1)<psplit
terms=c(3)

# Nodes 4,5
d=2
# Node 4
psplit=alpha*(1+d)^(-beta)
runif(1)<psplit
terms=c(terms,4)
# Node 5
psplit=alpha*(1+d)^(-beta)
runif(1)<psplit
terms=c(terms,5)
# Nowhere left to grow.


# Now select variable, cutpoints for internal nodes
# Since we have only 1 variable, its always used in splits
variables=rep(0,length(nonterms))

# Now get cuts
cutpoints=rep(0,length(nonterms))
cutpoints[1]=sample(cuts,1)
cutpoints[1]

# Now get cut for node 2
cuts=cuts[cuts<cutpoints[1]]
cutpoints[2]=sample(cuts,1)
cutpoints[2]


# Now draw terminal node parameters from N(0,tau^2)
tau2=1
mu=rep(0,length(terms))
for(i in 1:length(terms))
  mu[i]=rnorm(1,mean=0,sd=sqrt(tau2))


# Now plot the function represented by our tree
plot(c(0,cutpoints[2]),rep(mu[1],2),type='l',
     lwd=2,xlim=c(0,1),ylim=c(0,3),xlab="x",ylab="y")
lines(c(cutpoints[2],cutpoints[1]),rep(mu[2],2),lwd=2)
lines(c(cutpoints[1],1),rep(mu[3],2),lwd=2)




# Example using BayesTree
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


# load up the library
library(BayesTree)
preds=matrix(seq(0,1,length=100),ncol=1)

# Variance prior
shat=sd(y)
nu=3
q=0.90

# Mean prior
k=2

# Tree prior
alpha=0.95
beta=2
nc=100

# MCMC settings
N=1000
burn=1000


fit=bart(design,y,preds,sigest=shat,sigdf=nu,sigquant=q,
         k=k,power=beta,base=alpha,ntree=1,numcut=nc,
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



# Try nu=1 instead
nu=1
fit=bart(design,y,preds,sigest=shat,sigdf=nu,sigquant=q,
         k=k,power=beta,base=alpha,ntree=1,numcut=nc,
         ndpost=N,nskip=burn)

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



# Try nc=1000
nu=1
nc=1000
fit=bart(design,y,preds,sigest=shat,sigdf=nu,sigquant=q,
         k=k,power=beta,base=alpha,ntree=1,numcut=nc,
         ndpost=N,nskip=burn)

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


# Try nu=1, k=1
nu=1
k=1
nc=100
fit=bart(design,y,preds,sigest=shat,sigdf=nu,sigquant=q,
         k=k,power=beta,base=alpha,ntree=1,numcut=nc,
         ndpost=N,nskip=burn)

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

