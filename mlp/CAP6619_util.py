import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
import gzip
import os

from keras.utils.data_utils import get_file


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

def assignment3_load_data():
  """Loads the Fashion-MNIST dataset.
  # Returns
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  """
  dirname = os.path.join('datasets', 'fashion-mnist')
  base = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
  files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']

  paths = []
  for file in files:
    paths.append(get_file(file, origin=base + file, cache_subdir=dirname))

  with gzip.open(paths[0], 'rb') as lbpath:
    y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(paths[1], 'rb') as imgpath:
    x_train = np.frombuffer(imgpath.read(), np.uint8,
                                offset=16).reshape(len(y_train), 28, 28)

  with gzip.open(paths[2], 'rb') as lbpath:
    y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

  with gzip.open(paths[3], 'rb') as imgpath:
    x_test = np.frombuffer(imgpath.read(), np.uint8,
                               offset=16).reshape(len(y_test), 28, 28)

  return (x_train, y_train), (x_test, y_test)

  # plots training and validation loss
  # mac users may have to run with pythonw instead of python
def plot_losses(history):
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.show()
