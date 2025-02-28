import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Parameters
num_samples = 1000   # Number of data points
num_features = 2     # Number of features
num_classes = 2      # Number of classes (binary classification)

# Generate synthetic dataset
X, y = make_classification(
    n_samples=num_samples,
    n_features=num_features,
    n_informative=2,
    n_redundant=0,
    n_classes=num_classes,
    random_state=42
)

# Convert to DataFrame for better visualization
data = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(num_features)])
data['Label'] = y

# Save dataset to CSV
data.to_csv("Class_2/perceptron_dataset.csv", index=False)

# Plot the data (if 2D features)
plt.figure(figsize=(8, 6))
for label in np.unique(y):
    plt.scatter(X[y == label, 0], X[y == label, 1], label=f"Class {label}", alpha=0.7)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Random Dataset for Perception Learning")
plt.legend()
plt.show()

