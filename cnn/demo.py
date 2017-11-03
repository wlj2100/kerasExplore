#keras imports
from keras.datasets import mnist
from keras.models import Sequential
import keras
from keras.layers import Dense, Activation, convolutional, MaxPooling2D, Dropout, Flatten
from keras import optimizers
from keras import backend as K
K.set_image_dim_ordering('th')

# import useful functions for CAP6619
import CAP6619_util

# Global Parameters
# model choice 'cnn' to select a ConvNet, anything else defaults to mlp
model = ''
# batch size and number of training epochs
batch_size = 100
nb_epoch = 10
# data input dimensions (adjust to use other data such as cifar which would be 32,32,3)
img_rows, img_cols,img_channels = 28,28,1

# mlp model
def mlp_model():
  model = Sequential()
  model.add(Dense(128, input_dim=784))
  model.add(Activation('relu'))
  model.add(Dense(10))
  model.add(Activation('softmax'))
  return model

  # cnn model
def cnn_model():
  model = Sequential()
  model.add(convolutional.Conv2D(16, (3, 3),
                        input_shape=(img_channels, img_rows, img_cols)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D())
  model.add(Flatten())
  model.add(Dense(128))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(10))
  model.add(Activation('softmax'))
  return model

 # load mnist data, format for cnn model
def load_mnist_cnn():
  # load data
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  # reshape data (for cnn)
  x_train = x_train.reshape(x_train.shape[0], img_channels, img_rows, img_cols)
  x_test = x_test.reshape(x_test.shape[0], img_channels, img_rows, img_cols)
  # need categorical classes
  y_train = keras.utils.to_categorical(y_train, 10)
  y_test = keras.utils.to_categorical(y_test, 10)

  return x_train, y_train, x_test, y_test

 # load mnist data, format for mlp model
def load_mnist_mlp():
  # load data
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
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


# controls which model is used based on model selection at top of code
if (model == 'cnn'):
  x_train, y_train, x_test, y_test = load_mnist_cnn()
  model = cnn_model()
else:
  x_train, y_train, x_test, y_test = load_mnist_mlp()
  model = mlp_model()

# define optimizer
sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

# compile model
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# train model on training data
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch,
          verbose=1, validation_split=0.1)

# scores model on test data for chosen metric (accuracy)
score = model.evaluate(x_test, y_test, verbose=0)

# print accuracy
print(score[1])

# plots loss for training and validation data
cop6619_util.plot_losses(history)

# ends session, avoids potential error on program exit
K.clear_session()
