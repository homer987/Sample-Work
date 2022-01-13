import os
import sys
import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

# Change this to the location of the database directories
DB_DIR = os.path.dirname(os.path.realpath(__file__))

# Import databases
sys.path.insert(1, DB_DIR)
from db_utils import get_imdb_dataset, get_speech_dataset


class ReLU(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return tf.math.maximum(0.0, x)

# class leaky_relu(tf.keras.layers.Layer):
#     def __init__(self):
#         super().__init__()
#
#     def call(self, x):
#         return tf.math.maximum((x * 0.1), x)

############# I used tf.math.maximum, tf.math.minimum and tf.add functions. #############

class generic_vns_function(tf.keras.Model):

    def __init__(self, num_cnn_layers, filter_size, kernel_size, num_class=10):
        super().__init__()

        self.num_class = num_class
        for i in range (num_cnn_layers):
            self.cnn_layers.append(tf.keras.layers.Conv2D(filter_size[i], kernel_size[i], activation="relu"))
            #self.cnn_layers.append(tf.keras.layers.MaxPooling2D((2,2)))

        # Flatten
        self.flatten = tf.keras.layers.Flatten()
        # Dense layer
        self.dense1 = tf.keras.layers.Dense(1000, activation="relu")
        # add ReLU
        self.ReLU = ReLU()

        # add Leaky_relu
        #self.leaky_relu = leaky_relu()

        # output
        self.out = tf.keras.layers.Dense(num_class, activation="softmax")



    def call(self, x):
        for layer in self.layers:
            x = layer(x)

        tf.keras.layers.Flatten()
        tf.keras.layers.Dense(1000, activation="relu")

        return x


def train_model(model, epochs, batch_size, X_train, y_train, X_test, y_test):
    """Generic Deep Learning Model training function."""
    cb = [callbacks.EarlyStopping(monitor='val_loss', patience=3)]
    #model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs,
              #batch_size=batch_size, verbose=1, callbacks=cb)
    scores = model.evaluate(X_test, y_test, verbose=2)

    print("Baseline Error: %.2f%%" % (100-scores[1]*100))

    return model

def choose_dataset(dataset_type):
    """Select dataset based on string variable."""
    if dataset_type == "nlp":
        return get_imdb_dataset(dir=DB_DIR)
    elif dataset_type == "computer_vision":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset_type == "speech_recognition":
        (X_train, y_train), (X_test, y_test), (_, _) = get_speech_dataset()
    else:
        raise ValueError("Couldn't find dataset.")

    (X_train, X_test) = normalize_dataset(dataset_type, X_train, X_test)

    (X_train, y_train), (X_test, y_test) = reshape_dataset(X_train, y_train, X_test, y_test)

    return (X_train, y_train), (X_test, y_test)

def normalize_dataset(string, X_train, X_test):
    """Normalize speech recognition and computer vision datasets."""
    if string is "computer_vision":
        X_train = X_train / 255
        X_test = X_test / 255
    else:
        mean = np.mean(X_train)
        std = np.std(X_train)
        X_train = (X_train-std)/mean
        X_test = (X_test-std)/mean

    return (X_train, X_test)

def reshape_dataset(X_train, y_train, X_test, y_test):
    """Reshape Computer Vision and Speech datasets."""

    num_pixels = X_test.shape[1]*X_test.shape[2]

    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1).astype('float32')
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return (X_train, y_train), (X_test, y_test)

def main():

    # Hyperparameters
    layers = 3
    layer_units = 200
    epochs = 10
    batch_size = 50
    lr = 0.0001

    # Dataset : "computer_vision"
    dataset = "computer_vision"

    # Import Datasets
    (X_train, y_train), (X_test, y_test) = choose_dataset(dataset)

    # Generate and train model
    #model = generic_vns_function(X_train.shape[1], layers, y_train.shape[1], layer_units, lr).call(X_train.shape[1])
    opt = Adam(lr=lr)
    model = generic_vns_function(3, [128, 64, 32], [5, 5, 5])
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    trained_model = train_model(model, epochs, batch_size, X_train, y_train, X_test, y_test)
    #print(model.summary())

    # Save model to h5 file
    #trained_model.save('models/model_%s_a3.h5' % dataset, save_format='tf')

    return None

if __name__ == '__main__':
    main()
