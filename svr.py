# SVR

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('c:\\stock_market.csv')
X = dataset.iloc[:, 1:30].values

y = dataset.iloc[:, 30].values

#print(y)
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)
# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'linear')
regressor.fit(X, y)

y_pred=regressor.predict(X_test)
print(" The Predicted Output is \n",y_pred)
print(" The Actual Values are \n",y_test)
