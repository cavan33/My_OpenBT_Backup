# Here is the code I used to import things after a fresh R and RStudio install (or update).
# Also, I included my current system for changing working directories depending
# on if I'm doing research work or Stat 5730 work. This could be done with projects, 
# but I'm not bothering with that.


install.packages("remotes", dependencies=T)
library(remotes)
# the next one is included with remotes:
# install.packages("rmarkdown", dependencies=T)

install.packages("tidyverse", dependencies=T)
library(tidyverse)
# the next one is included with tidyverse, but I had to manually
# install to get tidyverse to work:
# install.packages("ggplot2", dependencies=T)

install.packages("xtable", dependencies=T)
install.packages("devtools", dependencies=T)
library(devtools) # Still can't get devtools to install, but that's OK

remotes::install_bitbucket("mpratola/openbt/Ropenbt")

# Also, working directory changers:
setwd("~/Documents/OpenBT/")
# setwd("~/Desktop/Stat 5730/RWorkDir_Stat5730")