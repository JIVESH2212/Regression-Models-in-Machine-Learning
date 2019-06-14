# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('c:\\stock_market.csv')
X = dataset.iloc[:, 1:26].values
y = dataset.iloc[:, 26:31].values
# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 9,random_state = 0)
yr=regressor.fit(X, y)
#print(yr)
# Predicting a new result
y_pred = regressor.predict(X)
print(" The Predicted Output is ",y_pred)
print(" The Actual Values are ",y)
