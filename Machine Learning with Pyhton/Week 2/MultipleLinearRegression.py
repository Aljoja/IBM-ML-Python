# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 14:29:19 2020

@author: Media
"""

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

# Reading the data in

df = pd.read_csv("FuelConsumptionCo2.csv")

# We have downloaded a fuel consumption dataset, FuelConsumption.csv, which contains model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada

# take a look at the dataset

df.head()

# Lets select some features that we want to use for regression.

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

# Lets plot Emission values with respect to Engine size:

plt.figure()
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Creating train and test dataset
"""
Train/Test Split involves splitting the dataset into training and testing sets respectively, which are mutually exclusive. 
After which, you train with the training set and test with the testing set. This will provide a more accurate evaluation on 
out-of-sample accuracy because the testing dataset is not part of the dataset that have been used to train the data. 
It is more realistic for real world problems.

This means that we know the outcome of each data point in this dataset, making it great to test with! 
And since this data has not been used to train the model, the model has no knowledge of the outcome of these data points. 
So, in essence, it’s truly an out-of-sample testing.
"""

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Train data distribution

plt.figure()
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Multiple Regression Model
"""
In reality, there are multiple variables that predict the Co2emission. When more than one independent variable is present, 
the process is called multiple linear regression. For example, predicting co2emission using FUELCONSUMPTION_COMB, EngineSize 
and Cylinders of cars. The good thing here is that Multiple linear regression is the extension of simple linear regression 
model.
"""

from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)

"""
As mentioned before, Coefficient and Intercept , are the parameters of the fit line. Given that it is a multiple linear 
regression, with 3 parameters, and knowing that the parameters are the intercept and coefficients of hyperplane, sklearn 
can estimate them from our data. Scikit-learn uses plain Ordinary Least Squares method to solve this problem.
"""

# Ordinary Least Squares (OLS)
"""
OLS is a method for estimating the unknown parameters in a linear regression model. OLS chooses the parameters of a linear 
function of a set of explanatory variables by minimizing the sum of the squares of the differences between the target 
dependent variable and those predicted by the linear function. In other words, it tries to minimizes the sum of squared 
errors (SSE) or mean squared error (MSE) between the target variable (y) and our predicted output ( 𝑦̂  ) over all samples 
in the dataset.

OLS can find the best parameters using of the following methods: - Solving the model parameters analytically using 
closed-form equations - Using an optimization algorithm (Gradient Descent, Stochastic Gradient Descent, Newton’s Method, etc.)
"""

# Prediction
y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f" % np.mean((y_hat - y) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))

# explained variance regression score:
"""
If  𝑦̂_hat is the estimated target output, y the corresponding (correct) target output, and Var is Variance, the square of 
the standard deviation, then the explained variance is estimated as follow:

explainedVariance(y, y_hat)= 1 - Var{y−y_hat}/Var{y}

The best possible score is 1.0, lower values are worse.
"""

# PRACTICE
"""
Try to use a multiple linear regression with the same dataset but this time use __FUEL CONSUMPTION in 
CITY__ and __FUEL CONSUMPTION in HWY__ instead of FUELCONSUMPTION_COMB. Does it result in better accuracy?
"""

# SOLUTION

regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
print ('Coefficients: ', regr.coef_)
y_= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f"% np.mean((y_ - y) ** 2))
print('Variance score: %.2f' % regr.score(x, y))

