import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
print(concrete_data.head())
print(concrete_data.shape)

concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

# Normalize the data by substracting the mean and dividing by the standard deviation

predictors_norm = (predictors - predictors.mean()) / predictors.std()
print(predictors_norm.head())

# Split the data into training and testing - 30% retained for testing
predictors_train, predictors_test, target_train, target_test = train_test_split(predictors_norm, target, test_size = 0.3, random_state = 42) # if you don't specify random state the random split will be different (random) every time you run your code - different results every time

print(predictors_train.shape)
print(predictors_test.shape)

# Save the number of predictors to *n_cols* since we will need this number when building our network
n_cols = predictors_norm.shape[1] # number of predictors
print(n_cols)

import keras
from keras.models import Sequential
from keras.layers import Dense

def regression_model():
    """
    function that defines our regression model for us so that we can conveniently call it to create our model.
    one hidden layer with 10 neurons
    """
    # create model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model