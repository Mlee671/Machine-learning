import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784) / 255.0
x_test  = x_test.reshape(10000, 784) / 255.0

print ("x_train shape:", x_train.shape)
print ("y_train shape:", y_train.shape)