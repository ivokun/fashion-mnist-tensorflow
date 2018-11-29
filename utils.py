import tensorflow as tf
from tensorflow import keras

import numpy as np

class training(object):

  def __init__(self, options):
    self.initializer = options.get('initializer')
    self.activation = options.get('activation')
    self.optimizer = options.get('optimizer')
    self.filterSize = options.get('filterSize')

  def model(self, imageTrain, convDense):
    fashionModel = keras.Sequential()
    for dense in range(convDense):
      fashionModel.add(keras.layers.Conv2D(self.filterSize*(dense+1), kernel_size = (3,3), padding= 'same', kernel_initializer = self.initializer, activation = self.activation, input_shape = imageTrain.shape[1:]))
      fashionModel.add(keras.layers.MaxPooling2D(2,2))
    fashionModel.add(keras.layers.Flatten())
    fashionModel.add(keras.layers.Dense(self.filterSize*(dense+1), self.activation))
    fashionModel.add(keras.layers.Dense(10, activation='softmax'))
    fashionModel.compile(optimizer= self.optimizer, loss='categorical_crossentropy',metrics=['accuracy'])

    return fashionModel

  def dataPreProcessing(self, data):
    data = data / 255

    meanSubt = np.mean(data)
    stdDev = np.std(data)

    data = data - meanSubt
    data = data / stdDev

    return data