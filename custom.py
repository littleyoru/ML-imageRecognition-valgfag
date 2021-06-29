#import numpy
import tensorflow as tf
from tensorflow import keras
from tensorflow import image
#from tensorflow.keras import layers
#from keras.layers import Dropout, Dense, Flatten, BatchNormalization
#from keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
#from keras.preprocessing.image import img_to_array
#from keras.constraints import maxnorm

import os
import sys
import json
import datetime
import numpy as np
import PIL
#from matplotlib import pyplot
#import skimage.draw
#import cv2
#import matplotlib.pyplot as plt


############################################################
#  Data Formatting
############################################################

def mold_image(images, config):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL

def mold_inputs(self, images):
    """Takes a list of images and modifies them to the format expected
    as an input to the neural network.
    images: List of image matrices [height,width,depth]. Images can have
        different sizes.

    Returns 3 Numpy matrices:
    molded_images: [N, h, w, 3]. Images resized and normalized.
    image_metas: [N, length of meta data]. Details about each image.
    windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
        original image (padding excluded).
    """
    molded_images = []
    image_metas = []
    windows = []
    for image in images:
        # Resize image
        # TODO: move resizing to mold_image()
        molded_image, window, scale, padding, crop = utils.resize_image(
            image,
            min_dim=32,
            min_scale=0,
            max_dim=800,
            mode="square")
        molded_image = mold_image(molded_image, self.config)
        # Build image_meta
        image_meta = compose_image_meta(
            0, image.shape, molded_image.shape, window, scale,
            np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
        # Append
        molded_images.append(molded_image)
        windows.append(window)
        image_metas.append(image_meta)
    # Pack into arrays
    molded_images = np.stack(molded_images)
    image_metas = np.stack(image_metas)
    windows = np.stack(windows)
    return molded_images, image_metas, windows

cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

newmodel = load_model("cifar10m.h5")

#optimizer = 'adam'
#newmodel.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

newmodel.load_weights("cifar10W.h5")
newmodel.summary()

# load photograph
img = load_img('cat6.jpg')
img = img_to_array(img)
#print(img)
print(img.shape)

# Mold inputs to format expected by the neural network
#newImg = tf.image.resize_with_pad(img, 32, 32, method=image.ResizeMethod.BILINEAR, antialias=False)
newImg = tf.image.resize_with_pad(img, 32, 32, method=image.ResizeMethod.NEAREST_NEIGHBOR, antialias=False)
print(newImg.shape)
print(newImg)

batch_image = np.zeros((1,) + newImg.shape, dtype=np.float32)
batch_image = batch_image.astype('float32')
batch_image = batch_image / 255
print(batch_image.shape[1:])
print(batch_image.shape)

# make prediction
results = newmodel.predict([batch_image], verbose=0)

highestConfidence = np.argmax(results[0])
print(cifar_classes[highestConfidence])

#print(results.shape)
#print(results)
results = results.round(2)
print(results)
for i in range(len(cifar_classes)):
    pr = 100*results[0][i]
    print(f'{pr.round(2)}% : {cifar_classes[i]}' )



#image_processed = 


        # # Callbacks
        # callbacks = [
        #     keras.callbacks.TensorBoard(log_dir=self.log_dir,
        #                                 histogram_freq=0, write_graph=True, write_images=False),
        #     keras.callbacks.ModelCheckpoint(self.checkpoint_path,
        #                                     verbose=0, save_weights_only=True),
        # ]

