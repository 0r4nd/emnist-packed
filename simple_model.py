# misc
import sys
import os

# load/save files
import json

# plot
import matplotlib.pyplot as plt
from PIL import Image

# datascience libs
import numpy as np
import math

from timeit import default_timer as timer

# datascience libs
import numpy as np
import pandas as pd

# tensorflow
from tensorflow import keras
import tensorflow as tf
#import tensorflowjs as tfjs

from keras import layers, models, optimizers
from keras.callbacks import EarlyStopping

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from emnist_loader import emnist_load_data



def set_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    #x = layers.Conv2D(32, (5,5), activation='relu', padding='same')(x)
    #x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (5,5), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(strides=(2,2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(strides=(2,2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs)


def compile_model(model):
    model.compile(optimizer = optimizers.Adam(lr=0.0001),
                  loss = 'categorical_crossentropy',
                  metrics = ["accuracy"])
    return model

def fit_model(model):
    callbacks = []
    es = EarlyStopping(
        monitor = 'val_accuracy',
        mode = 'max',
        min_delta = 0.001,
        patience = 10,
        restore_best_weights = True
    )
    callbacks.append(es)

    start_time = timer()
    history = model.fit(X_train,
                        y_train[0],
                        validation_data = (X_val, y_val[0]),
                        batch_size = 32,
                        epochs = 1000,
                        callbacks = callbacks,
                        verbose = 1)
    training_time = timer() - start_time
    print("Training time:", training_time)
    return history


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test), y_mapping = emnist_load_data("emnist-balanced", True)

    # convert target[0] to categorical
    num_classes = max(y_train[0])+1 # len(y_mapping)
    y_train[0] = to_categorical(y_train[0], num_classes = num_classes, dtype ="int8")
    y_test[0] = to_categorical(y_test[0], num_classes = num_classes, dtype ="int8")

    # add validation sets
    y_val = [None,None]
    (X_train, X_val) = train_test_split(X_train, test_size=0.25, random_state=1)
    (y_train[0], y_val[0]) = train_test_split(y_train[0], test_size=0.25, random_state=1)
    (y_train[1], y_val[1]) = train_test_split(y_train[1], test_size=0.25, random_state=1)

    # init model
    model = set_model(X_train[0].shape, num_classes)

    # compile
    compile_model(model)
    print(model.summary())

    # fit
    history = fit_model(model)

    # evaluate
    res = model.evaluate(X_val, y_val[0], verbose = 1)
    print(f'The accuracy on the val set is of {res[1]*100:.2f} %\n')
    res = model.evaluate(X_test, y_test[0], verbose = 1)
    print(f'The accuracy on the test set is of {res[1]*100:.2f} %')
