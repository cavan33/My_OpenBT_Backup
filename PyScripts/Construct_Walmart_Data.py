"""
Get the Walmart data from CSV to numpy arrays and/or pandas dataframes
"""
import pandas as pd
import numpy as np
# Example - the CO2 Plume data from Assignment 3
# Fit the model
walmart_pd = pd.read_csv('Documents/OpenBT/PyScripts/Walmart_Store_sales.csv')
x_pd = walmart_pd.drop(labels = "Weekly_Sales", axis = 1) # All columns except sales:
# Store, Date, Holiday Flag, Temperature, Fuel Price, CPI, Unemployment
y_pd = walmart_pd["Weekly_Sales"] # Weekly sales column

# Make these into numpy arrays:
x = x_pd.to_numpy()
y = y_pd.to_numpy()