from tensorflow import keras
from keras.callbacks import Callback
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime


class MyCallback(Callback):
    def __init__(self):
        super().__init__()
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))


def load_data():
    training_data = pd.read_csv('data_training.csv', header=None)
    training_label = pd.read_csv('labels_training.csv', header=None)
    testing_data = pd.read_csv('data_testing.csv', header=None)
    testing_label = pd.read_csv('labels_testing.csv', header=None)
    return training_data, training_label, testing_data, testing_label


def train_model(data):
    train_data, train_label, test_data, test_label = data
    lr = 0.05
    epochs = 32
    batch_size = 128
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    myCallback = MyCallback()

    fig_loss = plt.figure(figsize=(10, 10))
    fig_loss_ax = fig_loss.add_subplot(1, 1, 1)
    fig_acc = plt.figure(figsize=(10, 10))
    fig_acc_ax = fig_acc.add_subplot(1, 1, 1)

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(128, activation='relu', input_shape=(6,)))
    # model.add(keras.layers.Dense(64, activation='selu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
    history = model.fit(train_data, train_label, epochs=epochs, batch_size=batch_size, callbacks=[myCallback])

    results = {
        "per_epoch_training_accuracy": history.history["binary_accuracy"],
        "per_epoch_training_loss": history.history["loss"],
    }
    fig_loss_ax.plot(range(epochs), results["per_epoch_training_loss"], linestyle='solid',
                     label='training loss - lr %s' % lr)
    fig_acc_ax.plot(range(epochs), results["per_epoch_training_accuracy"], linestyle='solid',
                    label='testing acc - lr %s' % lr)

    fig_loss_ax.legend()
    fig_loss.show()

    fig_acc_ax.legend()
    fig_acc.show()
    return model


def main():
    start_time = datetime.datetime.now()
    data = load_data()
    print("Dataset loaded in %s!" % (datetime.datetime.now() - start_time))

    start_time = datetime.datetime.now()
    model = train_model(data)
    print("Model trained in %s!" % (datetime.datetime.now() - start_time))
    train_loss, train_acc = model.evaluate(data[0], data[1])
    test_loss, test_acc = model.evaluate(data[2], data[3])
    print("Training evaluation: ", train_loss, train_acc)
    print("Testing evaluation: ", test_loss, test_acc)


if __name__ == '__main__':
    tf.random.set_seed(45)
    np.random.seed(45)
    main()
