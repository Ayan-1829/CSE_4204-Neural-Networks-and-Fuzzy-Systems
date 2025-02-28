import numpy as np
import matplotlib.pyplot as plt

# Step 1: Initialize weights
def initialize_weights(grid_size, input_dim):
    return np.random.rand(grid_size, grid_size, input_dim)  # Random weights

# Step 2: Input pattern (Example: 2D points)
input_data = np.array([[1, 8], [5, 2], [9, 7], [4, 5], [1, 1], [2, 2], [3, 2], [3, 3]])

# SOM Parameters
grid_size = 3   # 3x3 grid of neurons
input_dim = 2   # Each input has 2 features
epochs = 70     # Number of iterations
learning_rate = 0.9
neighborhood_radius = 3

# Initialize weight matrix
weights = initialize_weights(grid_size, input_dim)

# Step 3: Find Best Matching Unit (BMU)
def find_bmu(weights, input_vector):
    distances = np.linalg.norm(weights - input_vector, axis=2)  # Euclidean distance
    bmu_index = np.unravel_index(np.argmin(distances), distances.shape)  # Find BMU
    return bmu_index

# Step 4: Update BMU and its neighbors
def update_weights(weights, bmu, input_vector, learning_rate, neighborhood_radius):
    grid_x, grid_y, _ = weights.shape
    for x in range(grid_x):
        for y in range(grid_y):
            distance_to_bmu = np.linalg.norm(np.array([x, y]) - np.array(bmu))
            if distance_to_bmu <= neighborhood_radius:
                # influence = np.exp(-distance_to_bmu**2 / (2 * (neighborhood_radius**2)))  # Gaussian function
                # weights[x, y] += influence * learning_rate * (input_vector - weights[x, y])

                weights[x, y] += learning_rate * (input_vector - weights[x, y])
    return weights

# Step 5: Update learning rate over time
def decay_learning_rate(initial_lr, epoch, t1= 500):
    return initial_lr * np.exp(-epoch / t1)

# Step 6: Update learning rate over time
def decay_neighbourhood(initial_lr, epoch, t2 = 500):
    return initial_lr * np.exp(-epoch / t2)

# Visualization
plt.figure(figsize=(6, 6))

# Initial state
plt.scatter(input_data[:, 0], input_data[:, 1], color='red', label='Input Data')
plt.scatter(weights[:, :, 0], weights[:, :, 1], color='blue', label='Neurons (Initial)')
plt.legend()
plt.title("Initial SOM Neuron Positions")
plt.show()

# Training Loop
for epoch in range(epochs):
    for input_vector in input_data:
        bmu = find_bmu(weights, input_vector)  # Step 3: Find BMU
        weights = update_weights(weights, bmu, input_vector, learning_rate, neighborhood_radius)  # Step 4: Update BMU
    learning_rate = decay_learning_rate(learning_rate, epoch)  # Step 5: Update learning rate
    neighborhood_radius = decay_neighbourhood(neighborhood_radius, epoch)  # Step 6: Update neighbourhood

    # Plot training progress every 10 epochs
    if epoch % 10 == 0 or epoch == epochs - 1:
        print("Learning rate: ", learning_rate)
        print("Neighbour radius: ", neighborhood_radius)
        print()

        plt.scatter(input_data[:, 0], input_data[:, 1], color='red', label='Input Data')
        plt.scatter(weights[:, :, 0], weights[:, :, 1], color='blue', label='Neurons')
        plt.title(f"SOM Training - Epoch {epoch}")
        plt.legend()
        plt.show()

# Final state
plt.scatter(input_data[:, 0], input_data[:, 1], color='red', label='Input Data')
plt.scatter(weights[:, :, 0], weights[:, :, 1], color='blue', label='Neurons (Final)')
plt.legend()
plt.title("Final SOM Neuron Positions")
plt.show()