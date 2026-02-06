import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import Button
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
    
epochs = 10
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

# Visualization and drawing interface
drawing = False

# Create empty 28x28 canvas
canvas = np.zeros((28, 28), dtype=np.float32)

fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(right=0.75)

# Show canvas
img = ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
ax.set_title("Draw a digit or test images using slider")
ax.axis("off")

# Probability text
prob_text = ax.text(
    1.05, 0.5, "", transform=ax.transAxes,
    fontsize=11, family="monospace",
    verticalalignment="center"
)

# Create a button axis (x, y, width, height)
ax_clear = plt.axes([0.8, 0.1, 0.15, 0.06])
clear_button = Button(ax_clear, "Clear")

# Add slider
ax_slider = plt.axes([0.25, 0.01, 0.5, 0.03])
slider = Slider(ax_slider, 'Index', 0, len(x_test)-1, valinit=0, valstep=1)




# Draw helper
def draw_pixel(x, y, radius=1):
    if 0 <= x < 28 and 0 <= y < 28:
                canvas[y, x] += 0.4  # white ink
                if canvas[y, x] > 1:
                    canvas[y, x] = 1
    for dx in range(-radius, radius+1):
        for dy in range(-radius, radius+1):
            xi = int(x + dx)
            yi = int(y + dy)
            if xi == x and yi == y:
                continue
            if 0 <= xi < 28 and 0 <= yi < 28:
                canvas[yi, xi] += 0.1  # white ink
                if canvas[yi, xi] > 1:
                    canvas[yi, xi] = 1

def update_prediction():
    input_img = canvas.reshape(1, 28, 28).flatten() 
    
    output = forward(input_img)[0].flatten()  # get output probabilities

    prob_lines = ["All output probabilities:"]
    for i, p in enumerate(output):
        prob_lines.append(f"{i} : {p*100:5.2f}%")
    prob_text.set_text("\n".join(prob_lines))

# Mouse events
def on_press(event):
    global drawing
    if event.inaxes == ax:
        drawing = True

def on_release(event):
    global drawing
    drawing = False

def on_move(event):
    if drawing and event.inaxes == ax:
        x = int(event.xdata)
        y = int(event.ydata)
        draw_pixel(x, y, radius=1)
        img.set_data(canvas)
        update_prediction()
        fig.canvas.draw_idle()

def clear_canvas(event):
    canvas[:] = 0.0   # white canvas (use 0.0 if black background)
    img.set_data(canvas)
    ax.set_title("Draw a digit or test images using slider")
    update_prediction()
    fig.canvas.draw_idle()

# Function to update image and predictions when slider moves
def update(val):
    global canvas
    image = x_test[int(slider.val)]
    # Normalize MNIST image to 0–1
    canvas[:] = image.astype(np.float32) / 255.0
    img.set_data(canvas)
    update_prediction()
    fig.canvas.draw_idle()

# Bind events
fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("motion_notify_event", on_move)
clear_button.on_clicked(clear_canvas)
slider.on_changed(update)

update_prediction()
fig.canvas.draw_idle()

plt.show()