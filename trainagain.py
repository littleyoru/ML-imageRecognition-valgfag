import numpy
import json
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dropout, Dense, Flatten, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.models import load_model


# Prepare data - input values are the pixels in the image
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# convert data from integer type to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# normalize input data from 0-255 to 0-1, to narrow the input values range, for better performance
x_train = x_train / 255
x_test = x_test / 255
# one-hot encode output
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_nr = y_test.shape[1]
print(class_nr)


newmodel = load_model("cifar10m.h5")
newmodel.summary()

# Training
epoch = 25
history = newmodel.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch, batch_size=64)

with open('data.json', 'w') as f:
    json.dump(history.history, f)

# Serialize weights to HDF5
newmodel.save_weights("cifar10Wnew.h5")

# Save model
newmodel.save("cifar10mnew.h5")