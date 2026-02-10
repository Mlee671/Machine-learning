import threading
import time
import numpy as np
import matplotlib.pyplot as plt

# Shared data
loss_history = []
training_done = False

def train_network(epochs=10):
    global training_done

    accuracy = 0.0
    for epoch in range(epochs):
        time.sleep(0.2)  # simulate training time

        # fake "learning"
        accuracy += 0.05 + np.random.uniform(-0.01, 0.01)
        accuracy = min(accuracy, 1.0)
        loss_history.append(accuracy)

        print(f"Epoch {epoch+1}, accuracy={accuracy:.4f}")

    training_done = True


# Start training in background
threading.Thread(target=train_network, daemon=True).start()

# --- Plot setup ---
#plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], marker="o")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Training in progress...")

# --- Live update loop ---
while not training_done or len(loss_history) == 0:
    plt.pause(0.1)

    if len(loss_history) > 0:
        x = np.arange(1, len(loss_history) + 1)
        y = loss_history

        line.set_data(x, y)
        ax.relim()
        ax.autoscale_view()
        plt.draw()

ax.set_title("Training complete")
#plt.ioff()
plt.show()