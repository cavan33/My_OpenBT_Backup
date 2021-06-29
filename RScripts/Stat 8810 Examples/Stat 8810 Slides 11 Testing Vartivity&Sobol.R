# Example using BayesTree, adapted to use OpenBT
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
n=10; rhotrue=0.2; lambdatrue=1
design=as.matrix(runif(n))
l1=list(m1=outer(design[,1],design[,1],"-"))
l.dez=list(l1=l1)
R=rhogeodacecormat(l.dez,c(rhotrue))$R+1e-5*diag(n)
L=t(chol(R))
u=rnorm(nrow(R))
z=L%*%u

# Our observed data:
y=as.vector(z)


# load up the OpenBT library (I used install_bitbucket to install it)
library(Ropenbt)

# Variance prior
# Now, set up the fit:
# Set values to be used in the fitting/predicting:
# Variance and mean priors
shat = 0.1 # (AKA shat) # in reality , it's lambdatrue^2?
# lower shat --> more fitting, I think
nu = 3
k = 2 # lower k --> more fitting, I think

# Tree prior
alpha = 0.95 # Default = 0.95
beta = 2 # Default = 2
nc = 1000 # (AKA numcut); Default = 100, but we usually use 1000

# MCMC settings
N = 1000 # (AKA ndpost); Default = 1000
burn = 1000 # (AKA nskip); Default = 100
nadapt = 1000 # default = 1000
tc = 4 # Default = 2, but we usually use 4
ntree = 200 # (AKA m); Default = 1
ntreeh = 1 # Default = 1

fit=openbt(design, y, tc=tc, pbd=c(0.7,0.0), overallsd=shat, overallnu = nu,
           ntreeh=ntreeh, ntree=ntree, k=k, model="bart", power=beta, base=alpha,
           numcut=nc, ndpost=N, nskip=burn, nadapt = nadapt)
str(fit)

# Save fit:
openbt.save(fit,"slides11fit_testv")

# Load fitted model to a new object (just for learning/testing purposes):
# fit2=openbt.load("slides11fit_testv")

# Calculate predictions:
preds=matrix(seq(0,1,length=100),ncol=1) # This is the x_test matrix
fitp=predict.openbt(fit, x.test=preds, tc=tc)
str(fitp)

# Easy part of the plot:
plot(design,y,pch=20,col="red",cex=2,xlim=c(0,1),
     ylim=c(-1.5,1),xlab="x",
     main="Predicted mean response +/- 2s.d.") # plot observed pts
# ^ Technically 1.96 SD (see fitp default quantiles)
# Plot the central line: Overall mean predictor
lines(preds,fitp$mmean,col="blue",xlab="observed",ylab="fitted", lwd=2)

# Full plot with the gray lines:
for(i in 1:length(fitp$mmean))
  lines(preds,fitp$mdraws[i,],col="grey",lwd=0.25)
print(length(fitp$mmean))
lines(preds,fitp$mmean-1.96*fitp$smean,lwd=0.75,col="black")
lines(preds,fitp$mmean+1.96*fitp$smean,lwd=0.75,col="black")
lines(preds,fitp$mmean,lwd=2,col="blue")
points(design,y,pch=20,col="red")

fitv = vartivity.openbt(fit)
# summarize_fitv(fitv)
# summarize_fitp(fitp)