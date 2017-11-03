import numpy as np
np.random.seed(1234)
from keras.utils import np_utils
import matplotlib.pyplot as plt
import gzip
import os

from keras.utils.data_utils import get_file

#keras imports
from keras.datasets import mnist
from keras.models import Sequential
import keras
from keras.layers import Dense, Activation, convolutional, MaxPooling2D, Dropout, Flatten
from keras import optimizers
from keras import backend as K

# import useful functions for CAP6619
import CAP6619_util

# part 1

# loads data for assignment 2
# returns 4 numpy arrays
# X_train: training set features
# Y_train: training set labels
# X_test: test set features
# Y_test: test set labels
def assignment2_load_data(data_dir):
  # change data_dir
  train = np.genfromtxt(data_dir+"fit.csv", delimiter=',')
  test = np.genfromtxt(data_dir+"test.csv", delimiter=',')
  X_train, y_train = train[:,:8], train[:,8:]
  X_test, y_test = test[:,:8], test[:,8:]
  X_train = X_train.astype('float32')
  X_test = X_test.astype('float32')
  Y_train = np_utils.to_categorical(y_train, 2)
  Y_test = np_utils.to_categorical(y_test, 2)
  return X_train, Y_train, X_test, Y_test

def part1():
  # get train/test set
  x_train, y_train, x_test, y_test = assignment2_load_data('./')

  # create mlp model
  model = Sequential()
  model.add(Dense(3, input_dim=8, activation='sigmoid'))
  model.add(Dense(2, activation='softmax'))
  model.summary()

  # define optimizer
  sgd = optimizers.SGD(lr=0.3, momentum=0.2, decay=0.0, nesterov=False)

  # compile model
  model.compile(loss='binary_crossentropy',
                optimizer=sgd,
                metrics=['accuracy'])

  # train model on training data
  history = model.fit(x_train, y_train, batch_size=100, epochs=500,
            verbose=0, validation_split=0.1)

  # scores model on test data for chosen metric (accuracy)
  score = model.evaluate(x_test, y_test, verbose=0)

  # print accuracy
  print(score[1])

  # plots loss for training and validation data
  CAP6619_util.plot_losses(history)

def part2():
  # get train/test set
  x_train, y_train, x_test, y_test = assignment2_load_data('./')

  # create mlp model
  model = Sequential()
  model.add(Dense(3, input_dim=8, activation='sigmoid'))
  model.add(Dense(3, activation='sigmoid'))
  model.add(Dense(2, activation='softmax'))
  model.summary()

  # define optimizer
  sgd = optimizers.SGD(lr=0.3, momentum=0.2, decay=0.0, nesterov=False)

  # compile model
  model.compile(loss='binary_crossentropy',
                optimizer=sgd,
                metrics=['accuracy'])

  # train model on training data
  history = model.fit(x_train, y_train, batch_size=100, epochs=500,
            verbose=0, validation_split=0.1)

  # scores model on test data for chosen metric (accuracy)
  score = model.evaluate(x_test, y_test, verbose=0)

  # print accuracy
  print(score[1])

  # plots loss for training and validation data
  CAP6619_util.plot_losses(history)

def part3():
  # get train/test set
  x_train, y_train, x_test, y_test = assignment2_load_data('./')

  # create mlp model
  model = Sequential()
  model.add(Dense(3, input_dim=8, activation='sigmoid'))
  model.add(Dense(2, activation='softmax'))
  model.summary()

  # define optimizer
  adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

  # compile model
  model.compile(loss='binary_crossentropy',
                optimizer=adam,
                metrics=['accuracy'])

  # train model on training data
  history = model.fit(x_train, y_train, batch_size=100, epochs=500,
            verbose=0, validation_split=0.1)

  # scores model on test data for chosen metric (accuracy)
  score = model.evaluate(x_test, y_test, verbose=0)

  # print accuracy
  print(score[1])

  # plots loss for training and validation data
  CAP6619_util.plot_losses(history)
if __name__ == '__main__':
    part2()
