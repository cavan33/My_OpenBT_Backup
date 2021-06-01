# Plot the density function of a standard normal:
xs = seq(-3,3,length=100)
y = dnorm(xs)
plot(xs,y, type="l")

# Example of plotting a density function using a bunch of samples 
# from sampling the dist. you know (without knowing the conceptual density function)
norm_sample=rnorm(1000)
plot(density(norm_sample))

# Trace plot
plot(norm_sample, type="l")

# Points version, to show that density is higher at the middle, as expected:
plot(norm_sample, type="p")
