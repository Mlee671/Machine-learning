import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import Button
import Mnist_neural_net as nn

data = np.load("assets/mnist.npz")
training_data = np.load(f"training_data/W_and_b(64, 32, 10) epoch(20).npz")


x_test  = data["x_test"]
y_test  = data["y_test"]

X_test  = x_test.reshape(x_test.shape[0], -1)


# Converts labels to one-hot encoding
def one_hot(y, num_classes=10):
    out = np.zeros((y.shape[0], num_classes))
    out[np.arange(y.shape[0]), y] = 1
    return out

Y_test  = one_hot(y_test)

# Initializes weights and biases for a 3-layer neural network
# Input layer: 784 neurons (28x28 pixels) connected to hidden layer 1 with 128 neurons
W1 = training_data["W1"]
b1 = training_data["b1"]

# Hidden layer: 128 neurons connected to hidden layer 2 with 64 neurons
W2 = training_data["W2"]
b2 = training_data["b2"]

# hidden layer 2: 64 neurons connected to output layer with 10 neurons (for 10 classes)
W3 = training_data["W3"]
b3 = training_data["b3"]

# Activation functions and loss function
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

# Softmax function for output layer
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Forward pass through the network
def forward(X):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = relu(z2)

    z3 = np.dot(a2, W3) + b3
    output = softmax(z3)

    return output
    
preds = forward(X_test)
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
    
    output = forward(input_img).flatten()  # get output probabilities

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