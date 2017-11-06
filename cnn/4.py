import numpy as np
#for reproducability
np.random.seed(1234)
from keras.utils import np_utils
import matplotlib.pyplot as plt
import gzip
import os
from keras.utils.data_utils import get_file

from keras.models import Sequential
import keras
from keras.layers import Dense, Activation, convolutional, MaxPooling2D, Dropout, Flatten
from keras import optimizers
from keras import backend as K
K.set_image_dim_ordering('th')

# globla var
img_rows, img_cols,img_channels = 28,28,1

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

def plot_losses(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

# mlp model
def mlp_model():
    model = Sequential()
    # 28*28 = 784
    model.add(Dense(128, input_dim=784, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

# cnn model
def cnn_model():
    model = Sequential()
    model.add(convolutional.Conv2D(filters=16, kernel_size=3, input_shape=(img_channels, img_rows, img_cols), activation='relu', strides=1
    ))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    # flatten to 1 dim
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'softmax'))
    return model

def cnn_model_1():
    model = Sequential()
    model.add(convolutional.Conv2D(filters=16, kernel_size=3, input_shape=(img_channels, img_rows, img_cols), activation='relu', strides=1
    ))
    model.add(convolutional.Conv2D(filters=16, kernel_size=3, activation='relu', strides=1
    ))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    # flatten to 1 dim
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'softmax'))
    return model

def cnn_model_2():
    model = Sequential()
    model.add(convolutional.Conv2D(filters=16, kernel_size=3, input_shape=(img_channels, img_rows, img_cols), activation='relu', strides=1
    ))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(convolutional.Conv2D(filters=16, kernel_size=3, activation='relu', strides=1
    ))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    # flatten to 1 dim
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'softmax'))
    return model

def cnn_model_3():
    model = Sequential()
    model.add(convolutional.Conv2D(filters=32, kernel_size=3, input_shape=(img_channels, img_rows, img_cols), activation='relu', strides=1
    ))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    # flatten to 1 dim
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'softmax'))
    return model

def cnn_model_4():
    model = Sequential()
    model.add(convolutional.Conv2D(filters=16, kernel_size=5, input_shape=(img_channels, img_rows, img_cols), activation='relu', strides=1
    ))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    # flatten to 1 dim
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'softmax'))
    return model

def cnn_model_5():
    model = Sequential()
    model.add(convolutional.Conv2D(filters=16, kernel_size=7, input_shape=(img_channels, img_rows, img_cols), activation='relu', strides=1
    ))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    # flatten to 1 dim
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'softmax'))
    return model

# cnn model
def cnn_model_6():
    model = Sequential()
    model.add(convolutional.Conv2D(filters=16, kernel_size=3, input_shape=(img_channels, img_rows, img_cols), activation='relu', strides=1
    ))
    # flatten to 1 dim
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'softmax'))
    return model

# cnn model
def cnn_model_7():
    model = Sequential()
    model.add(convolutional.Conv2D(filters=16, kernel_size=3, input_shape=(img_channels, img_rows, img_cols), activation='relu', strides=1
    ))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    # flatten to 1 dim
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    return model

def load_mnist_mlp():
    # load data
    (x_train, y_train), (x_test, y_test) = assignment3_load_data()
    # flatten data for mlp
    x_train = x_train.reshape(60000, img_rows*img_cols*img_channels)
    x_test = x_test.reshape(10000, img_rows*img_cols*img_channels)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    # need categorical classes
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test

# load mnist data, format for cnn model
def load_mnist_cnn():
    # load data
    (x_train, y_train), (x_test, y_test) = assignment3_load_data()
    # reshape data (for cnn)
    x_train = x_train.reshape(x_train.shape[0], img_channels, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], img_channels, img_rows, img_cols)
    # need categorical classes
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test

def part1():
    x_train, y_train, x_test, y_test = load_mnist_mlp()
    model = mlp_model()
    # define optimizer
    # adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # train model on training data
    history = model.fit(x_train, y_train, batch_size=100, epochs=10, verbose=1, validation_split=0.1)

    # scores model on test data for chosen metric (accuracy)
    score = model.evaluate(x_test, y_test, verbose=1)

    # print accuracy
    print()
    print(score[1])

    # plots loss for training and validation data
    plot_losses(history)


def part2():
    x_train, y_train, x_test, y_test = load_mnist_cnn()
    model = cnn_model_7()
    # define optimizer
    # adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # train model on training data
    history = model.fit(x_train, y_train, batch_size=100, epochs=10, verbose=1, validation_split=0.1)

    # scores model on test data for chosen metric (accuracy)
    score = model.evaluate(x_test, y_test, verbose=1)

    # print accuracy
    print '\n'
    print(score[1])

    # plots loss for training and validation data
    plot_losses(history)


if __name__ == '__main__':
    # part1()
    part2()
