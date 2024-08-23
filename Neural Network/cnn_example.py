# Here we test our CNN on an MNIST dataset

import numpy as np

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activation import tanh, tanh_prime
from losses import mse, mse_prime

import keras
from keras.datasets import mnist
from keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape and normalize data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10, (e.q x_3 = 3)
y_train = to_categorical(y_train)

x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = to_categorical(y_test)


net = Network()
net.add(FCLayer(28*28, 100))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(50, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(ActivationLayer(tanh, tanh_prime))

net.use(mse, mse_prime)
net.fit(x_train[0:1000], y_train[0:1000], epochs=30, learning_rate=0.1)

# test 
out = net.predict(x_test[0:5])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:5])