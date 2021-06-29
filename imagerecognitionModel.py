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


seed = 21
# Prepare data - input values are the pixels in the image
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# convert data from integer type to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# normalize input data from 0-255 to 0-1, to narrow the input values range, for better performance
x_train = x_train / 255
x_test = x_test / 255
print(x_test)
# one-hot encode output
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_nr = y_test.shape[1]
print(class_nr)
print(x_train.shape ,x_train.shape[1:])

# Definind the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:], activation='relu', padding='same'))
model.add(Dropout(0.2)) # drop 20% of the existing connections between layers
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2))) # downsampling to account for image distortions
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(128, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

# model.add(Dense(64, activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())

model.add(Dense(class_nr, activation='softmax'))

# Configure for training
epoch = 25
optimizer = 'adam'
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print(model.summary())

# Training
numpy.random.seed(seed)
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch, batch_size=64)
#history = model.fit(x_train, y_train, validation_split=0.1, epochs=epoch, batch_size=64)

with open('data.json', 'w') as f:
    json.dump(history.history, f)


# Serialize weights to HDF5
model.save_weights("cifar10W.h5")

# Load weights
#model.load_weights("cifar10W.h5")

# Save model
model.save("cifar10m.h5")