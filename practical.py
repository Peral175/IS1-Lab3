import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import Callback
from datetime import datetime
from sklearn.model_selection import train_test_split


### Implement this method!
#   HINT: make sure to return your "best" model
#       i.e. the model that gives you the best accuracy on testing data and the least over/under-fitting
#       (have a look at grade_model() function to know how we evaluate over/under-fitting!)
#       this will result in the best grade! (you may use the grade_model() function to verify)
#   HINT: In order to return the best model, the easiest way is to remember the weights for the best NN
#       and restore them before returning the model
#       (see model.get_weights & model.set_weights:https://keras.io/models/about-keras-models/)
#       further you are encouraged to apply callbacks (as in the exercises) to determine and retrieve the best model!
#       (https://keras.io/callbacks/)
#   HINT: you are not allowed to change code except for
#   HINT: make sure that your results are stable, we will recompute them ourselves using your code in order to grade you
#       you should apply random seed to achieve this (as already prepared, but it might not (yet) be enough...)

class MyCallback(Callback):
    def __init__(self):
        self.losses = []
        self.accuracies = []
    
    def on_batch_end(self,batch,logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))

def train_model(dataset):

    ## TODO implement your code HERE (and add callbacks above as you see need!)
    import matplotlib.pyplot as plt
    train_data, train_label, test_data, test_label = dataset
    
    learning_rate   = 0.0072
    epochs          = 80
    batch_size      = 18
    optimizer       = keras.optimizers.Nadam(learning_rate=learning_rate, beta_1=0.90, beta_2=0.999, epsilon=1e-07, name="Nadam")
    myCallback      = MyCallback()
    dropout         = 0.070

    fig_loss = plt.figure(figsize=(10,10))
    fig_loss_ax = fig_loss.add_subplot(1, 1, 1)

    fig_acc = plt.figure(figsize=(10,10))
    fig_acc_ax = fig_acc.add_subplot(1, 1, 1)
    
    model = tf.keras.models.Sequential()
    model.add(keras.layers.Dense(128,activation='elu',input_shape=(6,)))
    model.add(keras.layers.Dense(32,activation='selu'))
    if dropout is not None:
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(32,activation='softplus'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
    
    history = model.fit(train_data,train_label, epochs=epochs,batch_size=batch_size, verbose=1, callbacks=[myCallback])
    test_loss, test_acc = model.evaluate(test_data, test_label)   
    results = {
            "per_epoch_training_accuracy": history.history["binary_accuracy"],
            "per_epoch_training_loss": history.history["loss"],
            }
    fig_loss_ax.plot(range(epochs),results["per_epoch_training_loss"],linestyle='solid', label='training loss - lr %s' %learning_rate)
    fig_acc_ax.plot(range(epochs),results["per_epoch_training_accuracy"],linestyle='solid', label='testing acc - lr %s' %learning_rate)
    
    fig_loss_ax.set_ylim((0.0, 1.0))
    fig_loss_ax.legend()
    fig_loss.show()

    fig_acc_ax.set_ylim((0.80, 1.0))
    fig_acc_ax.legend()
    fig_acc.show()
    return model

# DO NOT MODIFY THIS CODE!
def load_dataset():
    training_data = pd.read_csv("dataset/data_training.csv", header=None)
    test_data = pd.read_csv("dataset/data_testing.csv", header=None)
    training_labels = pd.read_csv("dataset/labels_training.csv", header=None)
    test_labels = pd.read_csv("dataset/labels_testing.csv", header=None)

    print(training_data.describe())
    print(training_labels.describe())
    print(test_data.describe())
    print(test_labels.describe())

    return training_data, training_labels, test_data, test_labels


# DO NOT TRY TO CHANGE THIS METHOD! ;)
def grade_model(model, dataset, silent=False):
    lower = 0.75
    upper = 0.92
    scale = 20.0
    bonus = 2.0
    overfitting_margin = 0.5  # 0.5 percent are allowed!

    train_data, train_label, test_data, test_labels = dataset
    train_loss, train_acc = model.evaluate(train_data, train_label)
    test_loss, test_acc = model.evaluate(test_data, test_labels)
    grade = (test_acc - lower) / (upper - lower)
    overfitting = abs(test_acc - train_acc)
    overfitting_penalty = max(overfitting*100.0 - overfitting_margin, 0.0) * 0.5  # overfitting and underfitting will be punished by 0.5 point/percent
    grade = min(grade * scale, scale + bonus)      # you can get up to 2 points bonus for a high accuracy
    grade = max(grade - overfitting_penalty, 0.0)  # but it will be cut down if overfitting/underfitting is present!
    if not silent:
        print("Accuracy  -  test: %s; training: %s; overfitting: %s" % (test_acc, train_acc, overfitting))
        print("Grade: %s (/%s + %s)  (overfitting penalty: %s)" % (grade, scale, bonus, overfitting_penalty))
    return grade


# DO NOT MODIFY THIS CODE!
def main():
    start_time = datetime.now()
    random.seed(42)
    tf.random.set_seed(42)
    np.random.seed(42)
    dataset = load_dataset()
    print("dataset loaded %s" % (datetime.now() - start_time))
    start_time = datetime.now()

    model = train_model(dataset)  ##  we will use the model that you return in "train_model()" to grade you!
    grade_model(model, dataset)
    print("Done %s" % (datetime.now() - start_time))


if __name__ == "__main__":
    main()
