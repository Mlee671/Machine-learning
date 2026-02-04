import numpy as np

# creating two arrays containing values to test and expected result
test_array = np.array([[0,0],[0,1],[1,0],[1,1]])
expected_result = np.array([[0],[1],[1],[0]])

#parameter for learning rate
learning_rate = 0.1
epochs = 10000

# Layers
input_layer_neurons = 2
hidden_layer_neurons = 3
output_neurons = 1

# Initializing weights and biases
# np.random.seed(42)
hidden_weights = np.random.randn(input_layer_neurons, hidden_layer_neurons)
hidden_bias = np.zeros((1,hidden_layer_neurons))
output_weights = np.random.randn(hidden_layer_neurons,output_neurons)
output_bias = np.zeros((1,output_neurons))

# Activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

for epoch in range(epochs):
    # Forward Propagation
    # input to hidden layer
    hidden_layer_activation = np.dot(test_array,hidden_weights) + hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)

    # hidden layer to output layer
    output_layer_activation = np.dot(hidden_layer_output,output_weights) + output_bias
    predicted_output = sigmoid(output_layer_activation)

    # Backpropagation
    error = expected_result - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Updating Weights and Biases
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * learning_rate
    hidden_weights += test_array.T.dot(d_hidden_layer) * learning_rate
    hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * learning_rate
    
# Displaying final output after training
results = []
    
for data in predicted_output:
    held = data[0]
    if held >= 0.5:
        results.append(str(int(held*100)) + "% certainty of being TRUE")
    else:
        results.append(str(100 - int(held*100)) + "% certainty of being FALSE")

print("Final predicted output:")
for i in range(len(test_array)):
    print(f"Input: {test_array[i]} Predicted XOR: {results[i]}")