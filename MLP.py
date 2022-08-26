# sources
# https://towardsdatascience.com/multi-layer-perceptron-using-tensorflow-9f3e218a4809
# https://www.geeksforgeeks.org/multi-layer-perceptron-learning-in-tensorflow/
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
import matplotlib.pyplot as plt


def load_dataset():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    # normalize x
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_dataset()
    print('Dataset loaded:')
    print('Feature matrix:', X_train.shape)
    print('Target matrix:', X_test.shape)
    print('Feature matrix:', y_train.shape)
    print('Target matrix:', y_test.shape)
    # visualize dataset
    # fig, ax = plt.subplots(10, 10)
    # k = 0
    # for i in range(10):
    #     for j in range(10):
    #         ax[i][j].imshow(X_train[k].reshape(28, 28),
    #                         aspect='auto')
    #         k += 1
    # plt.show()
    # create our model (784 * 256 * 128 * 10)
    model = Sequential([
        # reshape 28 row * 28 column data to 28*28 rows
        Flatten(input_shape=(28, 28)),
        # dense layer 1
        Dense(512, activation='relu'),
        Dropout(.2),
        # dense layer 2
        Dense(256, activation='relu'),
        Dropout(.2),
        # output layer
        Dense(10, activation='softmax'),
    ])
    # compile with optimizer (Adam, SGD etc) and loss (MSE etc)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # train
    # epoch: how many times to do the batch training
    # batch: number of samples per epoch, next epoch trains on the next batch_size of data i.e. staircase
    # validation_split: how much of the training data to be used for loss evaluation
    # (model will not be trained on this data)
    model.fit(X_train, y_train, epochs=10,
              batch_size=600,
              validation_split=0.2)
    # results
    results = model.evaluate(X_test, y_test, verbose=0)
    print('test loss, test acc:', results)
    # 512 * 256, batch_size 2000, no dropouts: 0.9745
    # 512 * 256, batch_size 1000, no dropouts: 0.9785
    # 512 * 256, batch_size 800, no dropouts: 0.9789
    # 512 * 256, batch_size 800, dropout .2: 0.9803
    # 512 * 256, batch_size 600, dropout .2: 0.9819
    # 512 * 256 * 256, batch_size 600, dropout .2: 0.9822