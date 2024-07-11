import dnuLearn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate synthetic dataset
n_samples = 4000
n_components = 4

X, y_true = make_blobs(n_samples=n_samples, centers=n_components, cluster_std=0.60, random_state=0)

# Train the k-means model
kmean_model = dnuLearn.KMeanClustering()
center = kmean_model.train(X, 4)

# Print the centers
print(center)

# Plot the data points
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_true, s=10, cmap='viridis')

# Mark the cluster centers
plt.scatter(center[:, 0], center[:, 1], c='red', s=30, alpha=0.75, marker='X')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Generated Blobs with 4 Components and Cluster Centers')
plt.show()
