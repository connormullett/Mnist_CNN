
import keras
import tensorflow as tf
from keras.models import load_model

mnist = keras.datasets.mnist
(x, y), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape((10000, 28, 28, 1))


model = load_model('mnistmodel.h5')
loss, acc = model.evaluate(x_test, y_test)
print(acc, loss)

