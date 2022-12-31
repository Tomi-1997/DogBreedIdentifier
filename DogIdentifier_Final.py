CHANGES = 'Added droput'

## TODO : Class weights

import time
import pandas as pd # open csv
import numpy as np
import matplotlib.pyplot as plt
import os, random # get random dog batch
import cv2
from matplotlib.image import imread
from PIL import Image, ImageOps
import tensorflow as tf
from sklearn.model_selection import train_test_split ## Split data to train \ validate

## Model imports
from keras.models import Sequential  # initial NN
from keras.layers import Dense, Dropout, SpatialDropout2D, GlobalAveragePooling2D  # construct each layer
from keras.layers import Convolution2D # swipe across the image by 1
from keras.layers import MaxPooling2D  # swipe across by pool size
from keras.layers import ZeroPadding2D
from keras.layers import Flatten
from keras.losses import categorical_crossentropy
from keras.layers import RandomRotation, RandomFlip, RandomZoom
from keras import regularizers

DIR = 'C:\\Users\\tomto\\PycharmProjects\\RandomStuff\\dog-breed-identification\\'
TRAIN_DIR = DIR + '\\train\\'

DATA = pd.read_csv(DIR + '\\labels.csv')
BREEDS = pd.unique(DATA['breed'])
BREEDS_NUM = len(BREEDS)


DATA_NAMES = [fname for fname in DATA['id']]   ## All filenames
DATA_LABEL = [breed for breed in DATA['breed']] ## Breed to true/false vector matching their breed index in breeds list.
DATA_LABELS = [label == np.array(BREEDS) for label in DATA_LABEL] ## Apply to all.

## Values to resize each image to. (Should be low to take less ram)
IMG_HEIGHT = 100
IMG_WIDTH = IMG_HEIGHT
CHANNELS = 3

BATCH_SIZE = 32
EPOCH_NUM = 50

## Can be set to lower number for debugging
IMAGES = len(DATA)

def process_image(image_path):
    """
    This function will read image, resize the image and return into TF format.
    Arguments:
        image_path(str): path of image
    Returns:
        img: Tensor image
    """
    size = IMG_HEIGHT
    img = tf.io.read_file(image_path)


    # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
    img = tf.io.decode_image(img, channels = CHANNELS)
    # Convert the colour channel values from 0-225 values to 0-1 values
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (size, size))

    return img


def get_image_label(image_id, label):
    """
    Takes an image file name and the associated label,
    processes the image and returns a tuple of (image, label).
    """
    image_path = name_to_path(image_id)
    image = process_image(image_path)
    return image, label


def name_to_path(fname : str):
    return TRAIN_DIR + fname + '.jpg'


DATA_VALUE = [process_image(name_to_path(fname)) for fname in DATA_NAMES]


if __name__ == '__main__':
    # Split data into training & validation
    X_train, X_val, Y_train, Y_val = train_test_split(DATA_VALUE[:IMAGES],
                                                      DATA_LABELS[:IMAGES],
                                                      test_size=0.2, random_state=2)

    X_train = tf.convert_to_tensor(X_train)
    X_val = tf.convert_to_tensor(X_val)
    Y_train = tf.convert_to_tensor(Y_train)
    Y_val = tf.convert_to_tensor(Y_val)

    ## Sequential : Single inupt --> Single output (Image -> breed)
    model = Sequential(
        [
            RandomRotation(factor=0.2),
            RandomFlip(),
            RandomZoom(height_factor=(-0.2, 0.2))
        ]
    )

    reg = regularizers.l2(0.0001)
    activ = 'relu'

    ######################################### Model
    ## Condense information (conv2d).
    model.add(Convolution2D(filters = 32, kernel_size = (3), activation = activ,
                            kernel_regularizer = reg, input_shape = (IMG_HEIGHT,IMG_WIDTH,CHANNELS)))
    model.add(Convolution2D(64, (3, 3), activation = activ, kernel_regularizer = reg))
    model.add(Convolution2D(128, (3, 3), activation = activ, kernel_regularizer = reg))
    model.add(SpatialDropout2D(0.5)) ## Dropout certain pixels by groups
    model.add(GlobalAveragePooling2D()) ## Instead of Flatten()
    model.add(Dense(units = BREEDS_NUM, activation = 'softmax'))
    ######################################### Model

    model.compile(loss = categorical_crossentropy, optimizer = 'adam', metrics = ['accuracy'])
    history = model.fit(X_train, Y_train,
                        shuffle = True,
                        batch_size = BATCH_SIZE,
                        epochs = EPOCH_NUM,
                        verbose = 2, ## Print info
                        validation_data = (X_val, Y_val),
                        initial_epoch = 0)


    print(CHANGES)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(history.history['val_accuracy'], label='validation') ## Val
    ax1.plot(history.history['accuracy'], label='train')          ## Train

    ax2.plot(history.history['val_loss'], label='validation')
    ax2.plot(history.history['loss'], label='train')

    ax1.set_xlabel('epochs')
    ax1.set_title('accuracy')
    ax1.legend()
    ax2.set_xlabel('epochs')
    ax2.set_title('loss')
    ax2.legend()

    plt.show()
    plt.show()
    model.summary()