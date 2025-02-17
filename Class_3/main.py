import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# def sigmoid_derivative(x):
#     return x * (1 - x)

# dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(1)
input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.5

weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
bias_hidden = np.random.uniform(-1, 1, (1, hidden_size))
bias_output = np.random.uniform(-1, 1, (1, output_size))

epochs = 750
for epoch in range(epochs):
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)
    
    error = y - final_output
    
    d_output = error * sigmoid_derivative(final_output)
    d_hidden = d_output.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_output)
    
    weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
    
    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Error: {np.mean(np.abs(error))}')

print('\nTrained Model Outputs:')
predictions = []
for i in range(len(X)):
    hidden_input = np.dot(X[i], weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)
    predictions.append(final_output.round())
    print(f'Input: {X[i]}, Predicted Output: {final_output.round()}')

correct_predictions = np.sum(np.array(predictions).flatten() == y.flatten())
accuracy = (correct_predictions / len(y)) * 100
print(f'Accuracy: {accuracy:.2f}%')