import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset from CSV
data = pd.read_csv("Class_2/perceptron_dataset.csv")

# Split features and labels
X = data.iloc[:, :-1].values  # All columns except the last one
Y = data.iloc[:, -1].values   # Last column is the label

# Initialize weights
learning_rate = 0.1                            # Learning rate
iteration = 800                                 # Number of iterations
weights = np.random.rand(X.shape[1])            # One weight per feature
thresold = 1                                    # Thresold
weights = np.insert(weights, 0, -thresold)


# Perceptron learning algorithm 2
for iter in range(iteration):
    for i in range(len(X)):
        x = X[i]
        x = np.insert(x, 0, 1);
        weighted_sum = np.dot(x, weights)
        
        output = 1 if weighted_sum >= 0 else 0
        
        delta = Y[i] - output
        
        if Y[i]==0 and output==1:
            weights -= learning_rate*x
        elif Y[i]==1 and output==0:
            weights += learning_rate*x

bias = weights[0]
weights = weights[1:]

print("Bias:", end="\t")
print(bias)
print("Weights: ", end=" ")
print(weights)

# Testing the perceptron
correct_predictions = 0
for i in range(len(X)):
    weighted_sum = np.dot(X[i], weights) + bias
    output = 1 if weighted_sum >= 0 else 0
    if output == Y[i]:
        correct_predictions += 1

# Accuracy
accuracy = (correct_predictions / len(X)) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Plotting
if X.shape[1] == 2:  # Only plot if there are 2 features
    plt.figure(figsize=(8, 6))
    
    # Plot data points
    for label in np.unique(Y):
        plt.scatter(X[Y == label, 0], X[Y == label, 1], label=f"Class {label}", alpha=0.7)
    
    # Decision boundary: weights[0]*x1 + weights[1]*x2 + bias = 0
    # Rearrange to x2 = (-weights[0]/weights[1])*x1 - (bias/weights[1])
    x_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    if weights[1] != 0:  # Avoid division by zero
        y_values = (-weights[0] / weights[1]) * x_values - (bias / weights[1])
        plt.plot(x_values, y_values, color='red', label='Decision Boundary')
    else:
        print("Warning: Cannot plot decision boundary (weights[1] = 0).")
    
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Perceptron Decision Boundary")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Dataset has more than 2 features; decision boundary cannot be plotted.")
