import linear_regression as lr
import logistic_regression as logr
import KNearestNeighbhour as knn
import Kmeans 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


def generate_linear_data(num_points, num_features, noise_std):
    
    X = np.random.rand(num_points, num_features)
    true_coefficients = np.random.rand(num_features + 1)
    y = true_coefficients[0] + np.dot(X, true_coefficients[1:])
    noise = np.random.normal(0, noise_std, size=num_points)
    y += noise
    
    return X, y, true_coefficients

def generate_logistic_data(num_points, num_features, noise_std):
    
    X, y = make_classification(
        n_samples=num_points,
        n_features=num_features,
        n_informative=1,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        hypercube=False,
        class_sep=10,
        random_state=41
    )
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, X.shape)
        X += noise
    
    return X,y


num_points = 100
num_features = 2
noise_std = 0


# Generate data for linear Regression
'''
X, y, true_coefficients = generate_linear_data(num_points, num_features, noise_std)


model = lr.LinearRegression()
# print(model.solve(X,y,True))
model.gradDestSolve(X,y)
print('coef1 = ',model.getcoef()) #debugging
model.solve(X,y)
print('coef 2 = ',model.getcoef())

#Generate data for logistic Regression
X, y = generate_logistic_data(num_points, num_features, noise_std)


model = logr.LogisticRegression()
model.solvePerceptron(X,y)
bias,coeff = model.sloveGradientMethod(X,y)


# plt.scatter(X[:,0],X[:,1],c=y,cmap='winter',s=100)
# plt.show()
'''
n_samples = 300     
n_features = 2  
centers = 4           
cluster_std = 1.0     

X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()

# model = knn.Knn(X,y)
# pred = model.pridict(3,X_test)

# for y_hat,y in zip(pred,y_test):
#     print(f'y_hat = {y_hat} y = {y}')

model = Kmeans.KMeanClustering()
model.train(X_train,k=4)
model.plot_clusters(X_train, title='Training Data Clustering')

# Predict and visualize on test data
predictions = model.predict(X_test)
print("Predictions on test data:", predictions)
model.plot_clusters(X_test, labels=predictions, title='Test Data Clustering')