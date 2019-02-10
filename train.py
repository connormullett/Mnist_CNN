
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D

def main():
    mnist = keras.datasets.mnist

    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()

    train_data = train_data / 255.0
    test_data = test_data / 255.0

    train_data = train_data.reshape((60000, 28, 28, 1))
    test_data = test_data.reshape((10000, 28, 28, 1))

    model = Sequential([
        Conv2D(32, kernel_size=(5, 5),
                                activation='relu',
                                input_shape=(28, 28, 1)),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Conv2D(64, (5, 5), activation='relu', input_shape=(28, 28, 1)),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.4),
        Dense(10, activation='softmax')
    ])


    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_data, train_labels, epochs=5)
    model.evaluate(test_data, test_labels)
    model.save('mnistmodel.h5')


if __name__ == '__main__':
    main()

