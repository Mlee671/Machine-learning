import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import Button
import Mnist_neural_net as nn

Hidden1_size = 64
Hidden2_size = 32

lr = 0.2
epochs = 10

data = np.load("assets/mnist.npz")
try:
    training_data = np.load(f"training_data/nodes({Hidden1_size}, {Hidden2_size}) epoch({epochs}) lr({lr}).npz")
except FileNotFoundError:
    training_data = None

if training_data is None:
    print("Training data not found. Please wait while neural network trains first.")
    nn.train_data_set(Hidden1_size, Hidden2_size, epochs, lr)
    training_data = np.load(f"training_data/nodes({Hidden1_size}, {Hidden2_size}) epoch({epochs}) lr({lr}).npz")

# Load test data for images and preprocessing
x_test  = data["x_test"]
y_test  = data["y_test"]

# Preprocess test data
X_test  = x_test.reshape(x_test.shape[0], -1)
Y_test  = nn.one_hot(y_test)

# loads the weights and bias based of trained data
W1 = training_data["W1"]
b1 = training_data["b1"]

W2 = training_data["W2"]
b2 = training_data["b2"]

W3 = training_data["W3"]
b3 = training_data["b3"]

# tests accuracy
preds = nn.test_forward(X_test, W1, b1, W2, b2, W3, b3)
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

# Create a button axis (x, y, width, height)
ax_test = plt.axes([0.8, 0.1, 0.15, 0.06])
test_button = Button(ax_test, "test_now")
test_button.ax.set_visible(False)

# Create a button axis (x, y, width, height)
ax_train = plt.axes([0.8, 0.2, 0.15, 0.06])
training_button = Button(ax_train, "training view")

# Add slider
ax_slider = plt.axes([0.25, 0.01, 0.5, 0.03])
slider = Slider(ax_slider, 'Index', 0, len(x_test)-1, valinit=0, valstep=1)

# Slider axes (left, bottom, width, height)
ax_s1 = plt.axes([0.15, 0.50, 0.5, 0.03])
ax_s2 = plt.axes([0.15, 0.40, 0.5, 0.03])
ax_s3 = plt.axes([0.15, 0.30, 0.5, 0.03])
ax_s4 = plt.axes([0.15, 0.20, 0.5, 0.03])

s1 = Slider(ax_s1, "hidden_layer1", 0, valmax = 200, valstep = 1, valinit = Hidden1_size)
s2 = Slider(ax_s2, "hidden_layer2", 0, valmax = 100, valstep = 1, valinit = Hidden2_size)
s3 = Slider(ax_s3, "epochs", 0, valmax = 50, valstep = 1, valinit = epochs)
s4 = Slider(ax_s4, "lr", 0, valmax = 1, valstep = 0.01, valinit = lr)

def set_slider_visible(slider, visible):
    slider.ax.set_visible(visible)
    slider.label.set_visible(visible)
    slider.valtext.set_visible(visible)
    s.set_active(visible)

# function for Drawing
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
    
    output = nn.test_forward(input_img, W1, b1, W2, b2, W3, b3).flatten()  # get output probabilities

    prob_lines = ["All output\nprobabilities:"]
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

def change_view(event):
    
    training_button.label.set_text("drawing view" if slider.ax.get_visible() else "training view")
    
    for s in [s1, s2, s3, s4]:
        set_slider_visible(s, not s.ax.get_visible())
    
    for i in (slider, clear_button, test_button):
        i.set_active(not i.ax.get_visible())
        i.ax.set_visible(not i.ax.get_visible())
    
    for i in (img, prob_text):
        i.set_visible(not i.get_visible())
    
    fig.canvas.draw_idle()

def redraw(val):
    fig.canvas.draw_idle()

# Bind events
fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("motion_notify_event", on_move)
clear_button.on_clicked(clear_canvas)
training_button.on_clicked(change_view)
#test_button.on_clicked(run_neural_net)
slider.on_changed(update)
for s in [s1, s2, s3, s4]:
    set_slider_visible(s, False)
    s.on_changed(redraw)
    


update_prediction()
fig.canvas.draw_idle()

plt.show()

