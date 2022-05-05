# -*- coding: utf-8 -*-
"""
Created on Wed May  4 17:01:22 2022

@author: Alex
"""

from tensorflow import keras
from keras import models
from keras import layers
from keras.callbacks import Callback
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


# This is a sample callback which (uselessly) counts the number of batches and epochs that have been run
#  to show how callbacks can be used

class BatchAndEpochCountCallback(Callback):
    def __init__(self):
        super().__init__()
        self.batch_count = 0
        self.epoch_count = 0

    def on_batch_end(self, batch, logs={}):
        self.batch_count += 1

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_count += 1


# This callback class helps to capture the loss and accuracy after each batch
class PerBatchLossCallback(Callback):
    def __init__(self):
        super().__init__()
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))


# This callback class helps to calculate the loss and accuracy on the testing data after each epoch
#   the corresponding testing data is given on creation of the callback object
class TestAccCallback(Callback):
    def __init__(self, test_data):
        super().__init__()
        self.test_data = test_data
        self.test_acc = []
        self.test_loss = []

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        self.test_acc.append(acc)
        self.test_loss.append(loss)


# this function will divide the given array into "outputs" buckets and calculate the mean of those buckets
def smooth(arr, outputs):
    return [np.mean(c) for c in np.array_split(np.array(arr), outputs)]


def train(data, optimizer, epochs, batch_size, dropout=None):
    train_images, train_labels, test_images, test_labels = data

    model = models.Sequential()
    # linear stack of layers
    model.add(layers.Dense(128, activation='relu', input_shape=(28 * 28,)))
    if dropout is not None:
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    bec = BatchAndEpochCountCallback()
    ta = TestAccCallback((test_images, test_labels))
    pblc = PerBatchLossCallback()
    history = model.fit(train_images, train_labels, epochs=epochs,
                        batch_size=batch_size, callbacks=[bec, ta, pblc])

    # loss and accuracy on the testing set AFTER the LAST epoch
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print("Testing loss: %s" % test_loss)
    print("Testing Accuracy: %s" % test_acc)
    print("Summary: %s" % model.summary())

    results = {
        "epoch_count": bec.epoch_count,  # we access the data stored in the callback object,
        "batch_count": bec.batch_count,  # it was filled during the execution of "model.fit"
        "per_epoch_training_accuracy": history.history["accuracy"],
        "per_epoch_training_loss": history.history["loss"],
        "per_epoch_testing_accuracy": ta.test_acc,
        "per_batch_training_loss": pblc.losses,
        "per_batch_accuracy": pblc.accuracies,

    }

    return results


# loads and prepares the mnist dataset
def load_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_size = 10000  # default: 60000
    test_size = 2000  # default: 10000
    train_images = train_images[:train_size]
    train_labels = train_labels[:train_size]
    test_images = test_images[:test_size]
    test_labels = test_labels[:test_size]

    train_images = train_images.reshape((train_size, 28 * 28))
    train_images = train_images.astype('float32') / 255
    # we want to work with float in [0,1]
    test_images = test_images.reshape((test_size, 28 * 28))
    test_images = test_images.astype('float32') / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    #    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    #    train_images = train_images.reshape((60000,28*28))
    #    train_images = train_images.astype('float32')/255
    #    test_images = test_images.reshape((10000,28*28))
    #    test_images = test_images.astype('float32')/255
    #    train_labels = to_categorical(train_labels)
    #    test_labels = to_categorical(test_labels)

    return train_images, train_labels, test_images, test_labels


def reduce_train_size(data, size):
    train_images, train_labels, test_images, test_labels = data
    return train_images[:size], train_labels[:size], test_images, test_labels


FIGSIZE = (10, 10)


# Example
def example():
    # 4-tuple of (training image, training labels, testing image, testing labels)
    data = load_data()

    learning_rate = 0.005
    epochs = 3
    batch_size = 32

    fig = plt.figure(figsize=FIGSIZE)
    fig_ax = fig.add_subplot(1, 1, 1)
    optim = keras.optimizers.Adam(learning_rate=learning_rate)
    results = train(data, optim, epochs=epochs, batch_size=batch_size)
    print("During training we processed - epochs: %s, total batches: %s" %
          (results["epoch_count"], results["batch_count"]))

    color = next(fig_ax._get_lines.prop_cycler)['color']  # get "next" color to allow reuse for multiple "plots"
    fig_ax.plot(range(epochs), results["per_epoch_training_accuracy"], linestyle='solid', color=color,
                label='training acc')
    fig_ax.set_ylim((0.7, 1.0))

    fig_ax.legend()
    fig.show()


# Exercise 1
def ex1():
    # 4-tuple of (training image, training labels, testing image, testing labels)
    data = load_data()

    epochs = 5
    batch_size = 32
    learning_rates = [0.0001, 0.001, 0.005, 0.01, 0.1]

    # Ex1.2 + 1.3 plots
    fig_loss = plt.figure(figsize=FIGSIZE)
    fig_loss_ax = fig_loss.add_subplot(1, 1, 1)

    fig_acc = plt.figure(figsize=FIGSIZE)
    fig_acc_ax = fig_acc.add_subplot(1, 1, 1)

    #####################
    # Fill you code here (and extend the "train" method with callbacks and add entries to "result" as necessary)
    # Hint use here:
    # optimizer = keras.optimizers.Adam(learning_rate=lr, beta_1=0.90, beta_2=0.99, amsgrad=True)
    for lr in learning_rates:
        optim = keras.optimizers.Adam(learning_rate=lr, beta_1=0.90, beta_2=0.99, amsgrad=True)
        results = train(data, optim, epochs=epochs, batch_size=batch_size)

        fig_loss_ax.plot(range(epochs), results["per_epoch_training_loss"], linestyle='solid',
                         label='training loss - lr %s' % lr)
        fig_acc_ax.plot(range(epochs), results["per_epoch_testing_accuracy"], linestyle='solid',
                        label='testing acc - lr %s' % lr)
    #####################

    fig_loss_ax.set_ylim((0.0, 1.0))
    fig_loss_ax.legend()
    fig_loss.show()

    fig_acc_ax.set_ylim((0.75, 1.0))
    fig_acc_ax.legend()
    fig_acc.show()


def ex2():
    # 4-tuple of (training image, training labels, testing image, testing labels)
    data = load_data()

    epochs = 10
    batch_sizes = [4, 32, 128]
    learning_rate = 0.005
    optim = keras.optimizers.SGD(learning_rate=learning_rate)

    # Ex1.2 + 1.3
    fig_loss = plt.figure(figsize=FIGSIZE)
    fig_loss_ax = fig_loss.add_subplot(1, 1, 1)

    fig_acc = plt.figure(figsize=FIGSIZE)
    fig_acc_ax = fig_acc.add_subplot(1, 1, 1)

    #####################
    # Fill you code here (and extend the "train" method with callbacks and add entries to "result" as necessary)
    # Hint: reuse suitable (i.e. plotting) code from Ex1

    #    nr_of_batches = data[0]/batch_sizes
    for bs in batch_sizes:
        results = train(data, optim, epochs=epochs, batch_size=bs)

        fig_loss_ax.plot(smooth(results["per_batch_training_loss"], 1200), linestyle='solid',
                         label='training loss - bs %s' % bs)

        fig_acc_ax.plot(range(epochs), results["per_epoch_testing_accuracy"], linestyle='solid',
                        label='testing acc - bs %s' % bs)

    # fig_loss_ax.set_ylim((0.0, 1.0))
    fig_loss_ax.legend()
    fig_loss.show()

    # fig_acc_ax.set_ylim((0.75, 1.0))
    fig_acc_ax.legend()
    fig_acc.show()
    #####################


def ex3():
    # 4-tuple of (training image, training labels, testing image, testing labels)
    data = load_data()

    epochs = 25  # 50
    batch_size = 32
    learning_rate = 0.005
    optim = keras.optimizers.Adam(learning_rate=learning_rate)
    train_sizes = [1000, 5000, 10000, 30000, 60000]

    #####################
    # Fill you code here (and extend the "train" method with callbacks and add entries to "result" as necessary)
    # Hint: reuse suitable (i.e. plotting) code from previous exercises
    fig_loss = plt.figure(figsize=FIGSIZE)
    fig_loss_ax = fig_loss.add_subplot(1, 1, 1)

    fig_acc = plt.figure(figsize=FIGSIZE)
    fig_acc_ax = fig_acc.add_subplot(1, 1, 1)

    for dsize in train_sizes:
        dataset = reduce_train_size(data, dsize)
        results = train(dataset, optim, epochs=epochs, batch_size=batch_size)
        fig_loss_ax.plot(range(epochs), results["per_epoch_training_loss"], linestyle='solid',
                         label='training acc - training size %s' % dsize)
        fig_acc_ax.plot(range(epochs), results["per_epoch_testing_accuracy"], linestyle='solid',
                        label='testing acc - training size %s' % dsize)

    fig_loss_ax.set_ylim((0.0, 1.0))
    fig_loss_ax.legend()
    fig_loss.show()

    fig_acc_ax.set_ylim((0.75, 1.0))
    fig_acc_ax.legend()
    fig_acc.show()
    #####################


def ex4():
    # 4-tuple of (training image, training labels, testing image, testing labels)
    data = load_data()

    epochs = 25
    batch_size = 32
    learning_rate = 0.005
    optimizers = [keras.optimizers.SGD(learning_rate=learning_rate),
                  keras.optimizers.RMSprop(learning_rate=learning_rate),
                  keras.optimizers.Adam(learning_rate=learning_rate)]
    train_sizes = [1000, 30000]

    #####################
    # Fill you code here (and extend the "train" method with callbacks and add entries to "result" as necessary)
    # Hint: reuse suitable (i.e. plotting) code from previous exercises

    fig_loss = plt.figure(figsize=FIGSIZE)
    fig_loss_ax = fig_loss.add_subplot(1, 1, 1)

    fig_acc = plt.figure(figsize=FIGSIZE)
    fig_acc_ax = fig_acc.add_subplot(1, 1, 1)

    for optim in optimizers:
        dataset = reduce_train_size(data, optim)
        results = train(dataset, optim, epochs=epochs, batch_size=batch_size)
        fig_loss_ax.plot(range(epochs), results["per_epoch_training_loss"], linestyle='solid',
                         label='training acc - training size %s' % optim)
        fig_acc_ax.plot(range(epochs), results["per_epoch_testing_accuracy"], linestyle='solid',
                        label='testing acc - training size %s' % optim)

    fig_loss_ax.set_ylim((0.0, 1.0))
    fig_loss_ax.legend()
    fig_loss.show()

    fig_acc_ax.set_ylim((0.75, 1.0))
    fig_acc_ax.legend()
    fig_acc.show()

    #####################


def ex5():
    # 4-tuple of (training image, training labels, testing image, testing labels)
    data = load_data()

    epochs = 25
    batch_size = 32
    learning_rate = 0.005
    optim = keras.optimizers.Adam(learning_rate=learning_rate)
    train_sizes = [1000, 30000]
    dropouts = [None, 0.1, 0.5]

    #####################
    # Fill you code here (and extend the "train" method with callbacks and add entries to "result" as necessary)
    # Hint: reuse suitable (i.e. plotting) code from previous exercises

    #####################


if __name__ == "__main__":
    # example()
    # ex1()
    ex2()
#    ex3()   #change load_data() for this exercise
#    ex4()
#    ex5()
