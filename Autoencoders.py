
from sklearn import metrics
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split #split dataset into training and test data sets
from keras.tensorflow.models import Sequential, Model
from keras.tensorflow.layers import Dense, Activation, Input
from keras.tensorflow.utils import plot_model

epoch = 100

x = np.array([range(10)]).astype(np.float32)
print (x)

model = Sequential()
model.add(Dense(10, input_dim=x.shape[1], activation = 'relu'))
model.add(Dense(x.shape[1])) #multiple output neurons
model.compile (loss = 'mean_squared_error', optimizer = 'adam')
model.fit(x,x,verbose = 0, epochs = epoch)


pred = model.predict(x)
score = np.sqrt(metrics.mean_squared_error(pred,x))
print ("Score (RMSE): {}". format(score))
np.set_printoptions(supress = True)
print (pred)