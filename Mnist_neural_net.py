
import numpy as np

class MnistNeuralNet:
    
    def __init__(self, hidden1_size=64, hidden2_size=32, epochs=10, lr=0.1, data=None):
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.epochs = epochs
        self.lr = lr
        self.data = data if data is not None else self.load_mnist_data()
        print(self.hidden1_size, self.hidden2_size, self.epochs, f"{self.lr:.2f}")
        self.load_training_data_set()

    def set_parameters(self, node_layer1 : int, node_layer_2: int, new_epochs: int, learning_rate: float):
        self.hidden1_size = node_layer1
        self.hidden2_size = node_layer_2
        self.epochs = new_epochs
        self.lr = learning_rate

    def get_weights(self):
        return self.W1, self.b1, self.W2, self.b2, self.W3, self.b3

    def get_images(self):
        return self.data["x_test"]
        
    def get_parameters(self):
        return (self.hidden1_size, self.hidden2_size, self.epochs, self.lr)
    
    # Converts labels to one-hot encoding
    def one_hot(self, y, num_classes=10):
        out = np.zeros((y.shape[0], num_classes))
        out[np.arange(y.shape[0]), y] = 1
        return out

    # Activation functions and loss function
    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    # Softmax function for output layer
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    # Cross-entropy loss function
    def cross_entropy(self, pred, target):
        return -np.mean(np.sum(target * np.log(pred + 1e-9), axis=1))

    # Forward pass through the network
    def test_forward(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)

        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.relu(z2)

        z3 = np.dot(a2, self.W3) + self.b3
        output = self.softmax(z3)

        return output

    def train_new_data_set(self):

        batch_size = self.hidden1_size

        X_train = self.data["x_train"]  # Avoid modifying original data
        y_train = self.data["y_train"]  # Avoid modifying original labels
        x_test  = self.data["x_test"]   # Avoid modifying original test data

        # Reshapes the 28x28 images into 784-dimensional vectors
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = x_test.reshape(x_test.shape[0], -1)

        # Normalizes pixel values to the range [0, 1]
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0

        # transforms the test answers into one hot encoding
        y_train = self.one_hot(y_train)

        # Initializes weights and biases for a 3-layer neural network
        # Input layer: 784 neurons (28x28 pixels) connected to hidden layer 1 with 128 neurons
        self.W1 = np.random.randn(784, self.hidden1_size) * 0.01
        self.b1 = np.zeros((1, self.hidden1_size))

        # Hidden layer: 128 neurons connected to hidden layer 2 with 64 neurons
        self.W2 = np.random.randn(self.hidden1_size, self.hidden2_size) * 0.01
        self.b2 = np.zeros((1, self.hidden2_size))

        # hidden layer 2: 64 neurons connected to output layer with 10 neurons (for 10 classes)
        self.W3 = np.random.randn(self.hidden2_size, 10) * 0.01
        self.b3 = np.zeros((1, 10))

        for epoch in range(self.epochs):
            perm = np.random.permutation(len(X_train))
            X_train = X_train[perm]
            y_train = y_train[perm]

            total_loss = 0

            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                preds, cache = self.forward(X_batch)
                loss = self.cross_entropy(preds, y_batch)
                self.backward(cache, y_batch, self.lr)

                total_loss += loss

            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

        np.savez(
            f"training_data/nodes({self.hidden1_size}, {self.hidden2_size}) epoch({self.epochs}) lr({self.lr:.2f}).npz",
            W1 = self.W1,
            b1 = self.b1,
            W2 = self.W2,
            b2 = self.b2,
            W3 = self.W3,
            b3 = self.b3
        )

    def load_training_data_set(self):
        try:
            training_data = np.load(f"training_data/nodes({self.hidden1_size}, {self.hidden2_size}) epoch({self.epochs}) lr({self.lr:.2f}).npz")
        except FileNotFoundError:
            training_data = None

        if training_data is None:
            print("Training data not found. Please wait while neural network trains first.")
            self.train_new_data_set()
            training_data = np.load(f"training_data/nodes({self.hidden1_size}, {self.hidden2_size}) epoch({self.epochs}) lr({self.lr:.2f}).npz")
        else:
            print("Training data found. Loading weights and biases...")
            # loads the weights and bias based of trained data
            self.W1 = training_data["W1"]
            self.b1 = training_data["b1"]

            self.W2 = training_data["W2"]
            self.b2 = training_data["b2"]

            self.W3 = training_data["W3"]
            self.b3 = training_data["b3"]

        X_test = self.data["x_test"].reshape(self.data["x_test"].shape[0], -1).astype(np.float32) / 255.0
        Y_test = self.one_hot(self.data["y_test"])
        
        # tests accuracy
        preds = self.test_forward(X_test)
        accuracy = np.mean(np.argmax(preds, axis=1) == np.argmax(Y_test, axis=1))
        print(f"Test accuracy: {accuracy * 100:.2f}%")

# Forward pass through the network
    def forward(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)

        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.relu(z2)

        z3 = np.dot(a2, self.W3) + self.b3
        output = self.softmax(z3)

        cache = (X, z1, a1, z2, a2, output)
        return output, cache

    def backward(self, cache, y_true, lr):
        X, z1, a1, z2, a2, output = cache
        m = X.shape[0]

        # Output layer gradient
        dz3 = (output - y_true) / m
        dW3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        # Hidden layer 2
        da2 = dz3 @ self.W3.T
        dz2 = da2 * self.relu_deriv(z2)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Hidden layer 1
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_deriv(z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update weights
        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
