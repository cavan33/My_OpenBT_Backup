"""
I'm going to attempt to learn how pandas work, and how they can be applied to the arrays/dataframes
that we're getting from fit() and predict().
"""
import pandas as pd
import numpy as np
# W3 Schools Example:
mydataset = {
  'cars': ["BMW", "Volvo", "Ford"],
  'passings': [3, 7, 2]}
myvar = pd.DataFrame(mydataset)
print(myvar)
     
# Series is a 1D array holding data of any type. It can have labels, too.
a = [1, 7, 2]
myvar = pd.Series(a, index = ["x", "y", "z"])
print(myvar)
print(myvar['y'])
     
# You can use a key/value object to create a Series, too:
calories = {"day1": 420, "day2": 380, "day3": 390}
myvar = pd.Series(calories)
print(myvar)
     
# DataFrames are multi-dimensional tables:
data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]}
df = pd.DataFrame(data, index = ["day1", "day2", "day3"])
print(df) 
print(df.loc["day1"]) # Row index = "loc"; returns a Series
# Could do this is there were no labels: print(df.loc[[0,1]])