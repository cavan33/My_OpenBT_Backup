# Create a test function (Bran-in)
# Test Branin function, rescaled
braninsc <- function(xx)
{  
  x1 <- xx[1]
  x2 <- xx[2]
  
  x1bar <- 15*x1 - 5
  x2bar <- 15 * x2
  
  term1 <- x2bar - 5.1*x1bar^2/(4*pi^2) + 5*x1bar/pi - 6
  term2 <- (10 - 10/(8*pi)) * cos(x1bar)
  
  y <- (term1^2 + term2 - 44.81) / 51.95
  return(y)
}


# Simulate branin data for testing
set.seed(99)
n=500
p=2
x = matrix(runif(n*p),ncol=p)
y=rep(0,n)
for(i in 1:n) y[i] = braninsc(x[i,])

# Perform the fit:
library(Ropenbt)
fit=openbt(x,y,tc=4,model="bart",modelname="branin")

# Calculate in-sample predictions (predictions for x_test matrix)
fitp=predict.openbt(fit,x,tc=4)

# Make a simple plot
plot(y,fitp$mmean,xlab="observed",ylab="fitted")
abline(0,1)

# Save fitted model as test.obt in the working directory
openbt.save(fit,"test")

# Load fitted model to a new object.
fit2=openbt.load("test")

# Calculate variable activity information
fitv=vartivity.openbt(fit2)

# Plot variable activity
plot(fitv) # Seems to not work for me...
str(fitv)
boxplot(fitv$vdraws)

# Calculate Sobol indices
fits=sobol.openbt(fit2)
fits$msi
fits$mtsi
fits$msij

#For more examples, see example.R file in the repo

