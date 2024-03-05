# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 15:18:38 2020

@author: Media
"""

"""
In this notebook, we learn how to use scikit-learn for Polynomial regression. We download a dataset that is related to fuel 
consumption and Carbon dioxide emission of cars. Then, we split our data into training and test sets, create a model using 
training set, evaluate our model using test set, and finally use model to predict unknown value.
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

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
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
After which, you train with the training set and test with the testing set.
"""

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# POLYNOMIAL REGRESSION
"""
Sometimes, the trend of data is not really linear, and looks curvy. In this case we can use Polynomial regression methods. 
In fact, many different regressions exist that can be used to fit whatever the dataset looks like, such as quadratic, cubic, 
and so on, and it can go on and on to infinite degrees.

In essence, we can call all of these, polynomial regression, where the relationship between the independent variable x and 
the dependent variable y is modeled as an nth degree polynomial in x. Lets say you want to have a polynomial regression 
(let's make 2 degree polynomial):

    y = b + theta_1*x + theta_2*x^2
    
Now, the question is: how we can fit our data on this equation while we have only x values, such as Engine Size? 
Well, we can create a few additional features: 1, x and x^2.

PloynomialFeatures() function in Scikit-learn library, drives a new feature sets from the original feature set. 
That is, a matrix will be generated consisting of all polynomial combinations of the features with degree less than or equal 
to the specified degree. For example, lets say the original feature set has only one feature, ENGINESIZE. Now, if we select 
the degree of the polynomial to be 2, then it generates 3 features, degree=0, degree=1 and degree=2:
"""
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])


poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)
train_x_poly

"""
fit_transform takes our x values, and output a list of our data raised from power of 0 to power of 2 (since we set the degree
 of our polynomial to 2).

                    [v_1, v2, ... , v_n]' -----> [[1, v_1, v_1^2],
                                                  [1, v_2, v_2^2],
                                                  ..............,
                                                  [1, v_n, v_n^2]]
                    
in our example

                    [2, 2.4, 1.5 , ....]' -----> [[1, 2.,  4.],
                                                  [1, 2.4, 5.76],
                                                  [1, 1.5, 2.25],
                                                  ..............]
                    
It looks like feature sets for multiple linear regression analysis, right? Yes. It Does. Indeed, Polynomial regression is a 
special case of linear regression, with the main idea of how do you select your features. Just consider replacing the x with
x_1 , x_1, x_1^2  with x_2 , and so on. Then the degree 2 equation would be turn into:
    
    y = b + theta_1*x + theta_2*x^2
    
Now, we can deal with it as 'linear regression' problem. Therefore, this polynomial regression is considered to be a special
case of traditional multiple linear regression. So, you can use the same mechanism as linear regression to solve such a 
problems.

so we can use LinearRegression() function to solve it:
"""

clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly, train_y)
# The coefficients
print ('Coefficients: ', clf.coef_)
print ('Intercept: ',clf.intercept_)

"""
As mentioned before, Coefficient and Intercept , are the parameters of the fit curvy line. Given that it is a typical 
multiple linear regression, with 3 parameters, and knowing that the parameters are the intercept and coefficients of 
hyperplane, sklearn has estimated them from our new set of feature sets. Lets plot it:
"""
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")

# Evaluation
from sklearn.metrics import r2_score

test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )

# PRACTICE
"""
Try to use a polynomial regression with the dataset but this time with degree three (cubic). 
Does it result in better accuracy?
"""
poly3 = PolynomialFeatures(degree=3)
train_x_poly3 = poly3.fit_transform(train_x)
clf3 = linear_model.LinearRegression()
train_y3_ = clf3.fit(train_x_poly3, train_y)
# The coefficients
print ('Coefficients: ', clf3.coef_)
print ('Intercept: ',clf3.intercept_)
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf3.intercept_[0]+ clf3.coef_[0][1]*XX + clf3.coef_[0][2]*np.power(XX, 2) + clf3.coef_[0][3]*np.power(XX, 3)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")
test_x_poly3 = poly3.fit_transform(test_x)
test_y3_ = clf3.predict(test_x_poly3)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y3_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y3_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y3_ , test_y) )

