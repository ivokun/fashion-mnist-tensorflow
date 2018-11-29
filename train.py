import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedShuffleSplit
from keras import backend as k

import utils

import numpy as np
import matplotlib.pyplot as plt

(imageTrain, labelTrain), (imageTest, labelTest) = tf.keras.datasets.fashion_mnist.load_data()

plt.figure()
plt.imshow(imageTrain[0])
plt.grid(False)

labels = {0 : "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}

imageTrain = np.expand_dims(imageTrain, -1)
imageTest = np.expand_dims(imageTest, -1)

sss = StratifiedShuffleSplit(n_splits=5, random_state=0, test_size=1/6)
trainIndex, validIndex = next(sss.split(imageTrain, labelTrain))
imageValid, labelValid = imageTrain[validIndex], labelTrain[validIndex]
imageTrain, labelTrain = imageTrain[trainIndex], labelTrain[trainIndex]

labelTrain = keras.utils.to_categorical(labelTrain)
labelValid = keras.utils.to_categorical(labelValid)
labelTest = keras.utils.to_categorical(labelTest)

print("Fashion MNIST train -       rows: ", imageTrain.shape[0]," columns:", imageTrain.shape[1:4])
print("Fashion MNIST validation -  rows: ", imageValid.shape[0]," columns:", imageValid.shape[1:4])
print("Fashion MNIST test -        rows: ", imageTest.shape[0]," columns:", imageTest.shape[1:4])


"""
# Parameters

# Initializer parameter
*   **glorot_normal** for Xavier initialization
*   **he_normal** for He initialization

# Activation function:
*   **relu** for ReLU
*   **selu** for SELU

# Optimizer parameter:
*   **keras.optimizers.Adam()** for ADAM optimizer
*   **keras.optimizers.Adagrad()** for Adagrad optimizer
*   **keras.optimizers.RMSprop()** for RMSProp optimizer
*   **keras.optimizers.Adadelta()** for AdadeltaOptimizer()

# Filter size parameter: integer

# Epochs: integer

"""

parameters = {
    'initializer': "he_normal",
    'activation': keras.activations.relu,
    'optimizer': keras.optimizers.Adam(),
    'filterSize': 32
}


"""#Initiate Model"""

fashion = utils.training(parameters)

imageTrain = fashion.dataPreProcessing(imageTrain)
imageTest = fashion.dataPreProcessing(imageTest)
imageValid = fashion.dataPreProcessing(imageValid)

fashionModel = fashion.model(imageTrain, 3)

fashionModel.summary()
fashionTraining = fashionModel.fit(imageTrain, labelTrain, epochs = 10, verbose=1, validation_data=(imageValid, labelValid))

fashionTraining = fashionModel.fit(imageTrain, labelTrain, epochs=10, verbose=1, validation_data=(imageValid, labelValid))

test_loss, test_acc = fashionModel.evaluate(imageTest, labelTest)

print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

validLoss, validAccuracy = fashionModel.evaluate(imageValid, labelValid)

print('Valid Accuracy:', validAccuracy)
print('Valid Loss:', validLoss)

print("Accuracy:", fashionTraining.history['acc'][9])
print("Loss:", fashionTraining.history['loss'][9])
print("Validation Accuracy:", fashionTraining.history['val_acc'][9])
print("Validation Loss:", fashionTraining.history['val_loss'][9])
print("Test Accuracy:", test_acc)
print("Test Loss:", test_loss)


fashionModel.save('drive/My Drive/Master Life/cnn_fmnist_L2.h5')