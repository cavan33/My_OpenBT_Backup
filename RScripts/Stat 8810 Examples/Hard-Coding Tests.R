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

#--------------------------------------------------------
tc = 4
nslv=tc-1
ylist=split(y,(seq(n)-1) %/% (n/nslv))
ylist
