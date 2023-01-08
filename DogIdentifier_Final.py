import time
import pandas as pd # open csv
import numpy as np
import matplotlib.pyplot as plt
import os, random
import cv2
from matplotlib.image import imread
from PIL import Image, ImageOps
import tensorflow as tf
from sklearn.model_selection import train_test_split ## Split data to train \ validate

## Model imports
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Dropout, SpatialDropout2D, GlobalAveragePooling2D, ReLU
from keras.layers import Convolution2D as Conv2D, LeakyReLU # swipe across the image by 1
from keras.layers import MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

DIR = 'C:\\Users\\tomto\\PycharmProjects\\RandomStuff\\dog-breed-identification\\'
TRAIN_DIR = DIR + '\\train\\'

DATA = pd.read_csv(DIR + '\\labels.csv')
BREEDS = pd.unique(DATA['breed'])
BREEDS_NUM = len(BREEDS)
IMAGES = len(DATA)


##### Initialize data #####
DATA_NAMES = [fname for fname in DATA['id']]   ## All filenames
DATA_LABEL = [breed for breed in DATA['breed']] ## Breed to true/false vector matching their breed index in breeds list.
DATA_LABELS = [label == np.array(BREEDS) for label in DATA_LABEL] ## Apply to all.

IMG_HEIGHT = 64
IMG_WIDTH = IMG_HEIGHT
CHANNELS = 3

BATCH_SIZE = 16
EPOCH_NUM = 200

##### For each breed, assign a weight #####
num_of_each_breed = DATA['breed'].value_counts()
max_breed = max(num_of_each_breed)
class_weights = {i : (max_breed / num_of_each_breed[i]) for i in range(BREEDS_NUM)}

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

def name_to_path(fname : str):
    return TRAIN_DIR + fname + '.jpg'

# Filepath -> Open image -> To tensor
DATA_VALUE = [process_image(name_to_path(fname)) for fname in DATA_NAMES]

def guess_my_pics(model):
    for pic in os.listdir(DIR + "my_pics"):
        path = DIR + "my_pics\\" + pic
        test_img = [process_image(path)]
        res = model.predict(tf.convert_to_tensor(test_img))
        y_classes = res.argmax(axis=-1)
        print(f'Actual:{pic.replace(".jpg","")}, Guess:{BREEDS[y_classes]}')

def get_model():
    activ = ReLU()
    reg = regularizers.l2(0.001)
    drop_rate = 0.25
    filter_size = 8
    ## Sequential : Single inupt --> Single output (Image -> breed)
    model = Sequential()
    weight_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=4)
    bias_init = tf.keras.initializers.Zeros()
    
    ## Add Layers
    model.add(Conv2D(filter_size, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=weight_init,
                     kernel_regularizer=reg, bias_initializer=bias_init, input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)))
    model.add(activ)
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2, padding='same'))
    model.add(SpatialDropout2D(drop_rate))

    ## Add several layers, each time increase the filter size to extract different features.
    while filter_size < 128:
        filter_size *= 2
        model.add(Conv2D(filter_size, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=weight_init,
                         kernel_regularizer=reg, bias_initializer=bias_init))
        model.add(activ)
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=2, padding='same'))
        model.add(SpatialDropout2D(drop_rate))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(1200, activation=activ, kernel_initializer=weight_init,
                     kernel_regularizer=reg, bias_initializer=bias_init))

    model.add(Dropout(drop_rate))

    model.add(Dense(BREEDS_NUM, activation="softmax"))
    return model

def get_image_gen():
    return ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=40,  # randomly rotate images in the range 40 degrees
        zoom_range=0.2,  # Randomly zoom image 20%
        width_shift_range=0.2,  # randomly shift images horizontally 20%
        height_shift_range=0.2,  # randomly shift images vertically 20%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

def plot_history(history):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(history.history['val_accuracy'], label='validation')  ## Val
    ax1.plot(history.history['accuracy'], label='train')  ## Train

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

def train_model(model, callbacks, opt):
    model.compile(loss = categorical_crossentropy,
                  optimizer = opt,
                  metrics = ['accuracy'])
    return model.fit(datagen.flow(X_train, Y_train,
                           batch_size=BATCH_SIZE),
              shuffle=True,
              epochs=EPOCH_NUM,
              verbose=2,  ## Print info
              validation_data=(X_val, Y_val)
              , callbacks=callbacks
              , class_weight=class_weights)

if __name__ == '__main__':
    # Split data into training & validation
    X_train, X_val, Y_train, Y_val = train_test_split(DATA_VALUE[:IMAGES],
                                                      DATA_LABELS[:IMAGES],
                                                      test_size=0.2, random_state=2)

    X_train = tf.convert_to_tensor(X_train)
    X_val = tf.convert_to_tensor(X_val)
    Y_train = tf.convert_to_tensor(Y_train)
    Y_val = tf.convert_to_tensor(Y_val)

    Train = True
    Predict = False
    checkpoint_path = "training_final/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    model = get_model()

    if Train:
        
        datagen = get_image_gen() # Class which aguments data. (Rotates it, flips, zooms in)
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=15) # Stop training if validation dosen't increase for 15 epochs
        checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True, save_weights_only=True) # Save model
        datagen.fit(X_train)
        callbacks = [checkpointer, early_stopping] # Train model with these extra parameters.
        learning_rate = 0.001 # Start learning rate high and decrease
        while True:
        ###################################################### TRAIN
            print(f'Starting training with learning of {learning_rate}')
            history = train_model(model, callbacks, Adam(learning_rate=learning_rate))
            learning_rate /= 10
            
    if Predict:
        # Try to load earlier model, and predict pictures from my folder.
        try:
            model = tf.keras.models.load_model(checkpoint_path)
            guess_my_pics(model)
        except Exception:
            print(f'Could not load previous model.')
