import os
import numpy as np
import Mnist_neural_net as nn
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import Button

def show_screen(ai_model):
    global canvas, drawing

    drawing = False
    image_set = ai_model.get_images()

    # Create empty 28x28 canvas
    canvas = np.zeros((28, 28), dtype=np.float32)

    fig, ax = plt.subplots(figsize=(16, 6))
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
    ax_train = plt.axes([0.8, 0.2, 0.15, 0.06])
    training_button = Button(ax_train, "create_new_model")

    # Add slider
    ax_slider = plt.axes([0.2, 0.05, 0.5, 0.03])
    slider = Slider(ax_slider, 'Test image', 0, len(image_set)-1, valinit=0, valstep=1)

    # Slider axes (left, bottom, width, height)
    ax_s1 = plt.axes([0.72, 0.90, 0.2, 0.03])
    ax_s2 = plt.axes([0.72, 0.80, 0.2, 0.03])
    ax_s3 = plt.axes([0.72, 0.70, 0.2, 0.03])
    ax_s4 = plt.axes([0.72, 0.60, 0.2, 0.03])

    hidden_layer1, hidden_layer2, epochs, learning_rate = ai_model.get_parameters()
    s1 = Slider(ax_s1, "hidden_layer1", 1, valmax = 200, valstep = 1, valinit = hidden_layer1)
    s2 = Slider(ax_s2, "hidden_layer2", 1, valmax = 100, valstep = 1, valinit = hidden_layer2)
    s3 = Slider(ax_s3, "epochs", 1, valmax = 20, valstep = 1, valinit = epochs)
    s4 = Slider(ax_s4, "lr", 0.01, valmax = 1, valstep = 0.01, valinit = learning_rate)

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
        
        output = ai_model.test_forward(input_img).flatten()  # get output probabilities

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
        for s in [s1, s2, s3, s4]:
            s.reset()
        update_prediction()
        fig.canvas.draw_idle()

    # Function to update image and predictions when slider moves
    def update(val):
        image = image_set[int(slider.val)]
        # Normalize MNIST image to 0–1
        canvas[:] = image.astype(np.float32) / 255.0
        img.set_data(canvas)
        update_prediction()
        fig.canvas.draw_idle()

    def train(event):
        new_param = []
        for s in [s1, s2, s3, s4]:
            new_param.append(s.val)
            s.valinit = s.val  # Update valinit to current value for next reset

        ai_model.set_parameters(int(new_param[0]), int(new_param[1]), int(new_param[2]), new_param[3])
        cache.append(new_param)
        # Reload trained data
        ai_model.load_training_data_set()
        update(val=slider.val)  # Update the display with new model predictions
    
    # Bind events
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    slider.on_changed(update)
    clear_button.on_clicked(clear_canvas)
    training_button.on_clicked(train)

    update(val=0)  # Initial update to show first image and predictions
    fig.canvas.draw_idle()

    plt.show()
    
def Start():
    global cache
    print("Starting program...")
    cache = []
    
def Finish():
    print("Program finished.")
    for item in cache:
        file_path = f"training_data/nodes({item[0]}, {item[1]}) epoch({item[2]}) lr({item[3]:.2f}).npz"
        if os.path.isfile(file_path):
            os.remove(file_path) 
            
if __name__ == "__main__":
    #start up
    Start()
    
    ai_model = nn.MnistNeuralNet()
    show_screen(ai_model)
    
    Finish()