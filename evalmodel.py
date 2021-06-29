import numpy
from tensorflow import keras
from tensorflow.keras.models import Sequential
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
#from matplotlib import pyplot as plt

def plotLosses(history):  
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# Prepare data - input values are the pixels in the image
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(y_test)
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
newmodel.load_weights("cifar10W.h5")

#plotLosses(history)

#score = newmodel.evaluate(x_test, y_test, batch_size=128, verbose=0)
#print(model.metrics_names)
#print(score)


# Model evaluation
scores = newmodel.evaluate(x_test, y_test, verbose=0)
print(newmodel.metrics_names)
print("Accuracy: %.2f%%" % (scores[1]*100))