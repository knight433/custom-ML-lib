import dnuLearn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
import numpy as np

n_samples = 200
n_components = 3

X, y_true = make_blobs(n_samples=n_samples, centers=n_components,cluster_std=0.5, random_state=0)

kmean_model = dnuLearn.KMeanClustering(epoch=10)
center = kmean_model.train(X, n_components)
kmean_model._plot_training_progress(X,interval=1000)

n_samples = 200
n_components = 1

# def generate_linear_data(num_points, num_features, noise_std):
    
#     X = np.random.rand(num_points, num_features)
#     true_coefficients = np.random.rand(num_features + 1)
#     y = true_coefficients[0] + np.dot(X, true_coefficients[1:])
#     noise = np.random.normal(0, noise_std, size=num_points)
#     y += noise
    
#     return X, y, true_coefficients

# X, y, true_coefficients = generate_linear_data(n_samples, n_components,0.1)


# linear_model = dnuLearn.LinearRegression()
# linear_model.gradDestSolve(X, y, learningRate=0.1, interval=200,epoch=200)


# def generate_logistic_data(num_points, num_features, noise_std):
#     X, y = make_classification(
#         n_samples=num_points,
#         n_features=num_features,
#         n_informative=1,
#         n_redundant=0,
#         n_classes=2,
#         n_clusters_per_class=1,
#         hypercube=False,
#         class_sep=10,
#         random_state=41
#     )
#     if noise_std > 0:
#         noise = np.random.normal(0, noise_std, X.shape)
#         X += noise

#     return X, y

# # Example usage:
# n_samples = 100
# n_components = 2

# X, y = generate_logistic_data(n_samples, n_components, 0.2)
# logistic_model = dnuLearn.LogisticRegression()
# logistic_model.solveGradientMethod(X, y, learning_rate=0.1, epoch=100)