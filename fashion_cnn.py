import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

def dataPreProcessing(dataTrain, dataTest):
  dataTrain = dataTrain.reshape(dataTrain.shape[0], 28, 28, 1)
  dataTest = dataTest.reshape(dataTest.shape[0], 28, 28, 1)
  meanSubt = np.mean(dataTrain)
  stdDev = np.std(dataTrain)

  dataTrain = dataTrain - meanSubt
  dataTrain = dataTrain / stdDev

  dataTest = dataTest - meanSubt
  dataTest = dataTest / stdDev
  return dataTrain, dataTest

def trainingModel(input, options):
  fashionModel = keras.Sequential()
  initializerParam = "he_normal"
  fashionModel.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), kernel_initializer = initializerParam, padding='same', activation = 'relu', input_shape=(28,28,1)))
  fashionModel.add(keras.layers.MaxPooling2D(2,2))
  fashionModel.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), kernel_initializer = initializerParam, padding='same', activation = 'relu'))
  fashionModel.add(keras.layers.MaxPooling2D(2,2))
  fashionModel.add(keras.layers.Flatten())
  fashionModel.add(keras.layers.Dense(256, activation='relu'))
  fashionModel.add(keras.layers.Dense(10, activation='softmax'))
  fashionModel.compile(optimizer=tf.train.AdamOptimizer(), loss='categorical_crossentropy',metrics=['accuracy'])
  fashionTraining = fashionModel.fit(imageTrain, labelTrain, epochs=10, verbose=1, validation_split=0.2)
  return fashionTraining

(imageTrain, labelTrain), (imageTest, labelTest) = tf.keras.datasets.fashion_mnist.load_data()

labelTrain = keras.utils.to_categorical(labelTrain)
labelTest = keras.utils.to_categorical(labelTest)

labels = {0 : "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}

fashionModel = trainingModel(dataTrain)

test_loss, test_acc = fashionModel.evaluate(imageTest, labelTest)

print('Test accuracy:', test_acc)
print('Test loss:', test_loss)
