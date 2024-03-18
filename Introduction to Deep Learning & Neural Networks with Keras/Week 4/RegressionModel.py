import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


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

# build the model
model = regression_model()

# fit the model - this trains and test at the same time
model.fit(predictors_norm, target, epochs=50) #, validation_split=0.3, epochs=100, verbose=2)

predicted_targets = model.predict(predictors_test)

print(type(predicted_targets))
print(type(target_test))
print(predicted_targets.shape)
print(target_test.shape)
# print(predicted_targets[0:10])
# print(target_test.head(10))

# Convert pandas Series to numpy array
target_test_array = target_test.values
print(type(target_test_array))

# Compute mean squared error
mse = mean_squared_error(target_test_array, predicted_targets)

# Optionally, you can also compute the root mean squared error (RMSE)
rmse = np.sqrt(mse)

print(mse)

A = 0 