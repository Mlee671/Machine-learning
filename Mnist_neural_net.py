import numpy as np

data = np.load("assets/mnist.npz")

# Converts labels to one-hot encoding
def one_hot(y, num_classes=10):
    out = np.zeros((y.shape[0], num_classes))
    out[np.arange(y.shape[0]), y] = 1
    return out

# Activation functions and loss function
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

# Softmax function for output layer
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Cross-entropy loss function
def cross_entropy(pred, target):
    return -np.mean(np.sum(target * np.log(pred + 1e-9), axis=1))

def set_parameters(node_layer1, node_layer_2, new_epochs, learning_rate):
    global hidden1_size, hidden2_size, epochs, lr
    hidden1_size = node_layer1
    hidden2_size = node_layer_2
    epochs = new_epochs
    lr = learning_rate

# Forward pass through the network
def test_forward(X, W1, b1, W2, b2, W3, b3):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = relu(z2)

    z3 = np.dot(a2, W3) + b3
    output = softmax(z3)

    return output

def train_data_set():
    global W1, b1, W2, b2, W3, b3

    batch_size = hidden1_size

    X_train = data["x_train"]
    y_train = data["y_train"]
    x_test  = data["x_test"]
    y_test  = data["y_test"]

    # Reshapes the 28x28 images into 784-dimensional vectors
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = x_test.reshape(x_test.shape[0], -1)

    # Normalizes pixel values to the range [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # transforms the test answers into one hot encoding
    y_train = one_hot(y_train)


    # Initializes weights and biases for a 3-layer neural network
    # Input layer: 784 neurons (28x28 pixels) connected to hidden layer 1 with 128 neurons
    W1 = np.random.randn(784, hidden1_size) * 0.01
    b1 = np.zeros((1, hidden1_size))

    # Hidden layer: 128 neurons connected to hidden layer 2 with 64 neurons
    W2 = np.random.randn(hidden1_size, hidden2_size) * 0.01
    b2 = np.zeros((1, hidden2_size))

    # hidden layer 2: 64 neurons connected to output layer with 10 neurons (for 10 classes)
    W3 = np.random.randn(hidden2_size, 10) * 0.01
    b3 = np.zeros((1, 10))

    # Forward pass through the network
    def forward(X):
        z1 = np.dot(X, W1) + b1
        a1 = relu(z1)

        z2 = np.dot(a1, W2) + b2
        a2 = relu(z2)

        z3 = np.dot(a2, W3) + b3
        output = softmax(z3)

        cache = (X, z1, a1, z2, a2, output)
        return output, cache

    def backward(cache, y_true, lr):
        global W1, b1, W2, b2, W3, b3
        X, z1, a1, z2, a2, output = cache
        m = X.shape[0]

        # Output layer gradient
        dz3 = (output - y_true) / m
        dW3 = a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        # Hidden layer 2
        da2 = dz3 @ W3.T
        dz2 = da2 * relu_deriv(z2)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # Hidden layer 1
        da1 = dz2 @ W2.T
        dz1 = da1 * relu_deriv(z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update weights
        W3 -= lr * dW3
        b3 -= lr * db3
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    for epoch in range(epochs):
        perm = np.random.permutation(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]

        total_loss = 0

        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            preds, cache = forward(X_batch)
            loss = cross_entropy(preds, y_batch)
            backward(cache, y_batch, lr)

            total_loss += loss

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    np.savez(
        f"training_data/nodes({b1.size}, {b2.size}) epoch({epochs}) lr({lr}).npz",
        W1 = W1,
        b1 = b1,
        W2 = W2,
        b2 = b2,
        W3 = W3,
        b3 = b3
    )

def load_training_data():
    try:
        training_data = np.load(f"training_data/nodes({hidden1_size}, {hidden2_size}) epoch({epochs}) lr({lr}).npz")
    except FileNotFoundError:
        training_data = None

    if training_data is None:
        print("Training data not found. Please wait while neural network trains first.")
        train_data_set()
        training_data = np.load(f"training_data/nodes({hidden1_size}, {hidden2_size}) epoch({epochs}) lr({lr}).npz")

    # loads the weights and bias based of trained data
    W1 = training_data["W1"]
    b1 = training_data["b1"]

    W2 = training_data["W2"]
    b2 = training_data["b2"]

    W3 = training_data["W3"]
    b3 = training_data["b3"]

    X_test = data["x_test"].reshape(data["x_test"].shape[0], -1).astype(np.float32) / 255.0
    Y_test = one_hot(data["y_test"])
    
    # tests accuracy
    preds = test_forward(X_test, W1, b1, W2, b2, W3, b3)
    accuracy = np.mean(np.argmax(preds, axis=1) == np.argmax(Y_test, axis=1))
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    return W1, b1, W2, b2, W3, b3

def get_weights():
    return W1, b1, W2, b2, W3, b3

def get_images():
    return data["x_test"]
    
def get_parameters():
    return (hidden1_size, hidden2_size, epochs, lr)
