import numpy as np
from tensorflow.keras.datasets import mnist

(X_train, y_train), (x_test, y_test) = mnist.load_data()

np.savez(
    "mnist.npz",
    x_train=X_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test
)