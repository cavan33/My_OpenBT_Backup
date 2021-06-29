get_walmart_data <- function(){
  library(lubridate)
  walmart <- read.csv("Walmart_Store_sales.csv")
  x <- subset(walmart, select = c(1,2,4,5,6,7,8))
  x$Date <- as.Date(x$Date, format = "%d-%m-%Y")
  y <- subset(walmart, select = 3)
  # Convert the date row into an integer:
  # We'll use 1-1-2010 as day 1, since the data has only 2010 to 2012 data:
  # Also, add a "month" attribute:
  x$Month <- NA
  temp <- NA
  x <- x[c("Store", "Month", "Date", "Holiday_Flag", "Temperature",
           "Fuel_Price", "CPI", "Unemployment")] #re-ordering columns by name
  for (i in 1:length(x[, 1])){
    x$Month[i] <- month(x$Date[i])
    temp[i] <- x$Date[i] - as.Date("2009-12-31", format = "%Y-%m-%d")
  }
  x$Date <- temp
  # Strip the metadata:
  y2 <- y[, 1]
  return (list("x" = x, "y" = y2, "y_df" = y))
}