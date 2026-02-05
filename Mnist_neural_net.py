import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tensorflow.keras.datasets import mnist

(X_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshapes the 28x28 images into 784-dimensional vectors
X_train = X_train.reshape(X_train.shape[0], -1)
X_test  = x_test.reshape(x_test.shape[0], -1)


# Normalizes pixel values to the range [0, 1]
X_train = X_train.astype(np.float32) / 255.0
X_test  = X_test.astype(np.float32) / 255.0

# Converts labels to one-hot encoding
def one_hot(y, num_classes=10):
    out = np.zeros((y.shape[0], num_classes))
    out[np.arange(y.shape[0]), y] = 1
    return out

y_train = one_hot(y_train)
Y_test  = one_hot(y_test)

# selects the random seed for reproducibility
np.random.seed(42)

# Initializes weights and biases for a 3-layer neural network
# Input layer: 784 neurons (28x28 pixels) connected to hidden layer 1 with 128 neurons
W1 = np.random.randn(784, 128) * 0.01
b1 = np.zeros((1, 128))

# Hidden layer: 128 neurons connected to hidden layer 2 with 64 neurons
W2 = np.random.randn(128, 64) * 0.01
b2 = np.zeros((1, 64))

# hidden layer 2: 64 neurons connected to output layer with 10 neurons (for 10 classes)
W3 = np.random.randn(64, 10) * 0.01
b3 = np.zeros((1, 10))

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

def backward(cache, y_true, lr=0.01):
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
    
epochs = 15
batch_size = 64
lr = 0.01

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
    
preds, _ = forward(X_test)
accuracy = np.mean(np.argmax(preds, axis=1) == np.argmax(Y_test, axis=1))
print(f"Test accuracy: {accuracy * 100:.2f}%")

# change to test different images max : 10000
index = 0

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 5))
plt.subplots_adjust(bottom=0.25)  # leave space for slider

# Display the first image
image_display = ax.imshow(x_test[index], cmap='gray')
ax.set_title(f"True: {y_test[index]}")
ax.axis('off')


# Create a text object to display probabilities
prob_text = ax.text(1.05, 0.5, '', transform=ax.transAxes, fontsize=10,
                    verticalalignment='center', family='monospace')

# Add slider
ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
slider = Slider(ax_slider, 'Index', 0, len(x_test)-1, valinit=index, valstep=1)

# Function to update image and predictions when slider moves
def update(val):
    index = int(slider.val)
    image = x_test[index]
    image_display.set_data(image)
    ax.set_title(f"True: {y_test[index]}")

    # get output probabilities
    output = preds[index]  # get the output for the selected image
    # Format probabilities as text
    prob_lines = ["All output probabilities:"]
    for i, p in enumerate(output):
        prob_lines.append(f"{i} : {p*100:5.2f}%")
    prob_text.set_text("\n".join(prob_lines))

    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()