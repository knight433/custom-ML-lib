import numpy as np
import matplotlib.pyplot as plt
from collections import Counter   
from matplotlib.animation import FuncAnimation, FFMpegWriter

class LinearRegression:
    def __init__(self):
        self.coef = []

    def getcoef(self):
        return self.coef

    def predict(self, input_par):
        w0 = self.coef[0]
        weights = self.coef[1:]
        pred = np.dot(input_par, weights) + w0
        return pred

    def plotLR(self, input_ar, input_labels, coefficients):
        X = np.hstack([np.ones((input_ar.shape[0], 1)), input_ar])
        if input_ar.shape[1] == 1:
            plt.scatter(input_ar, input_labels, color='blue', label='Data points')
            predicted_labels = np.dot(X, coefficients)
            plt.plot(input_ar, predicted_labels, color='red', label='Regression line')
            plt.xlabel('Feature X')
            plt.ylabel('Target y')
            plt.title('Linear Regression Fit')
            plt.legend()
            plt.show()
        elif input_ar.shape[1] == 2:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(input_ar[:, 0], input_ar[:, 1], input_labels, color='blue', label='Data points')
            x_surf, y_surf = np.meshgrid(np.linspace(input_ar[:, 0].min(), input_ar[:, 0].max(), 100), 
                                         np.linspace(input_ar[:, 1].min(), input_ar[:, 1].max(), 100))
            z_surf = coefficients[0] + coefficients[1] * x_surf + coefficients[2] * y_surf
            ax.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.5, rstride=100, cstride=100)
            ax.set_xlabel('Feature X1')
            ax.set_ylabel('Feature X2')
            ax.set_zlabel('Target y')
            ax.set_title('3D Linear Regression Fit')
            ax.legend()
            plt.show()
        else:
            print("Plotting is only supported for 1 or 2 features.")

    def solve(self, input_ar, input_labels, plot=False):
        X = np.hstack([np.ones((input_ar.shape[0], 1)), input_ar])
        X_trans = np.transpose(X)
        part1 = np.dot(X_trans, X)
        part1_inv = np.linalg.inv(part1)
        self.coef = np.dot(np.dot(part1_inv, X_trans), input_labels)
        if plot:
            self.plotLR(input_ar, input_labels, self.coef)

    def gradDestSolve(self, input_ar, input_labels, learningRate=0.1, interval=500,epoch=50):
        weights = np.random.rand(input_ar.shape[1])
        w0 = 2
        n = len(input_labels)
        loss_history = []
        history = []

        for count in range(epoch):
            pred = np.dot(input_ar, weights) + w0
            loss = np.mean((input_labels - pred) ** 2)
            loss_history.append(loss)
            dL_w0 = -2 * np.sum(input_labels - pred) / n
            dl_wi = -2 * np.dot((input_labels - pred), input_ar) / n
            weights -= learningRate * dl_wi
            w0 -= learningRate * dL_w0
            history.append((w0, weights.copy()))

        self.coef = [w0] + list(weights)
        self._animateLR(input_ar, input_labels, history, interval)

    def _animateLR(self, input_ar, input_labels, history, interval):
        fig, ax = plt.subplots()
        def update(epoch):
            ax.clear()
            w0, weights = history[epoch]
            coef = [w0] + list(weights)
            X = np.hstack([np.ones((input_ar.shape[0], 1)), input_ar])
            predicted_labels = np.dot(X, coef)
            ax.scatter(input_ar, input_labels, color='blue', label='Data points')
            ax.plot(input_ar, predicted_labels, color='red', label='Regression line')
            ax.set_xlabel('Feature X')
            ax.set_ylabel('Target y')
            ax.set_title(f'Linear Regression Fit at Epoch {epoch + 1}')
            ax.legend()
        ani = FuncAnimation(fig, update, frames=len(history), interval=interval, repeat=False)
        plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class LogisticRegression:
    def __init__(self):
        self.weights = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def step(self, z):
        return 1 if z > 0 else 0

    def _plotLine(self, bias, coeff, X, y, ax):
        if len(coeff) == 2:
            m = -(coeff[0] / coeff[1])
            b = -(bias / coeff[1])
            x_in = np.linspace(-3, 3, 100)
            y_in = m * x_in + b
            ax.plot(x_in, y_in, 'r-')
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', s=100)
            ax.set_ylim(-3, 2)

    def predict(self, X):
        X = np.hstack([1, X])
        predictions = self.sigmoid(np.dot(X, self.weights))
        return (predictions >= 0.5).astype(int)

    def solvePerceptron(self, x_par, y_labels, learning_rate=0.1, epoch=1000):
        X = x_par
        x_par = np.insert(x_par, 0, 1, axis=1)
        self.weights = np.random.randn(x_par.shape[1])
        history = []
        indices = []

        for i in range(epoch):
            j = np.random.randint(0, x_par.shape[0])
            y_hat = self.step(np.dot(x_par[j], self.weights))
            self.weights = self.weights + learning_rate * (y_labels[j] - y_hat) * x_par[j]
            history.append(self.weights.copy())
            indices.append(j)

        bias = self.weights[0]
        weights = self.weights[1:]
        self._animatePerceptron(X, y_labels, history, indices)
        return bias, weights
    
    def solveGradientMethod(self, x_par, y_labels, learning_rate=0.1, epoch=1000):
        X = np.insert(x_par, 0, 1, axis=1)
        self.weights = np.ones(X.shape[1])
        history = []

        for _ in range(epoch):
            y_hat = self.sigmoid(np.dot(X, self.weights))
            self.weights = self.weights + learning_rate * (np.dot((y_labels - y_hat), X) / X.shape[0])
            history.append(self.weights.copy())

        bias = self.weights[0]
        weights = self.weights[1:]
        self._animatePerceptron(x_par, y_labels, history, list(range(len(history))))
        return bias, weights

    def _animatePerceptron(self, X, y, history, indices):
        fig, ax = plt.subplots()

        def update(epoch):
            ax.clear()
            weights = history[epoch]
            bias = weights[0]
            coeff = weights[1:]
            self._plotLine(bias, coeff, X, y, ax)
            ax.scatter(X[indices[epoch], 0], X[indices[epoch], 1], color='red', s=100, edgecolor='k', label='Point k')
            ax.set_title(f'Perceptron at Epoch {epoch + 1}')
            ax.legend()

        ani = FuncAnimation(fig, update, frames=len(history), interval=200, repeat=False)
        plt.show()


    
class Knn:
    
    def __init__(self,data,labels):
        
        self.data = data
        self.lables = labels

    def _distanceCal(self,x1,x2):
        
        return np.sqrt(np.sum((x1-x2)**2))
    
    def _showGraph(self,X,y):
        
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.show()

    def pridict(self,k,newPoint):
        
        self.k = k
        predictions = [self._predict(x) for x in newPoint]
        return np.array(predictions)

    def _predict(self, x):

        distances = [self._distanceCal(x, x_train) for x_train in self.data]
        k_indices = np.argsort(distances)[:self.k]
        
        k_nearest_labels = [self.lables[i] for i in k_indices]
        
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

class KMeanClustering:
    def __init__(self, epoch=100):
        self.epoch = epoch
        self.centers = None
        self.history = []

    def _distanceCal(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _plot_training_progress(self, data, interval=200):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        def update(epoch):
            ax.clear()
            labels = self.history[epoch][1]
            centers = self.history[epoch][0]
            unique_labels = np.unique(labels)
            colors = plt.cm.get_cmap('tab10', len(unique_labels))
            
            for i, label in enumerate(unique_labels):
                cluster_points = data[labels == label]
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, c=[colors(i)], label=f'Cluster {label}')
                
            ax.scatter(centers[:, 0], centers[:, 1], s=200, c='red', marker='X', label='Centers')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_title(f'Clusters and Centroids at Epoch {epoch + 1}')
            ax.legend()
        
        ani = FuncAnimation(fig, update, frames=len(self.history), interval=interval, repeat=False)
        plt.show()
        
    def train(self, data, k):
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        self.centers = np.random.uniform(low=min_vals, high=max_vals, size=(k, data.shape[1]))

        for epoch in range(self.epoch):
            cluster_labels = []
            for x in data:
                distances = [self._distanceCal(x, center) for center in self.centers]
                cluster = np.argmin(distances)
                cluster_labels.append(cluster)
            
            cluster_labels = np.array(cluster_labels)

            new_centers = np.zeros_like(self.centers)
            for i in range(k):
                points_in_cluster = data[cluster_labels == i]
                if points_in_cluster.size > 0:
                    new_centers[i] = np.mean(points_in_cluster, axis=0)
                else:  # If a cluster loses all points, reinitialize its center randomly
                    new_centers[i] = np.random.uniform(low=min_vals, high=max_vals, size=data.shape[1])
    
            self.centers = new_centers
            self.history.append((self.centers.copy(), cluster_labels.copy()))
        
        self.y = cluster_labels
        return self.centers

    def predict(self, data):
        predictions = []
        for x in data:
            distances = [self._distanceCal(x, center) for center in self.centers]
            cluster = np.argmin(distances)
            predictions.append(cluster)
        return np.array(predictions)
    
    def plot_clusters(self, data, labels=None):
        if labels is None:
            labels = self.predict(data)

        plt.figure(figsize=(8, 6))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap('tab10', len(unique_labels))

        for i, label in enumerate(unique_labels):
            cluster_points = data[labels == label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, c=[colors(i)], label=f'Cluster {label}')

        plt.scatter(self.centers[:, 0], self.centers[:, 1], s=200, c='red', marker='X', label='Centers')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()

class svm:

    def __init__(self) -> None:
        self.w = None
        self.b = None
    
    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

    def plot_svm(self,X,y):
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        x1_1 = get_hyperplane_value(x0_1, self.w, self.b, 0)
        x1_2 = get_hyperplane_value(x0_2, self.w, self.b, 0)

        x1_1_m = get_hyperplane_value(x0_1, self.w, self.b, -1)
        x1_2_m = get_hyperplane_value(x0_2, self.w, self.b, -1)

        x1_1_p = get_hyperplane_value(x0_1, self.w, self.b, 1)
        x1_2_p = get_hyperplane_value(x0_2, self.w, self.b, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])

        plt.show()

    def solve (self,x,y,learningRate=0.01,lambda_para = 0.01,epcoh=1000):
        
        n_samples, n_features = x.shape
        
        y_ = np.where(y <= 0,-1,1) # assigning -ve or +ve part of the plains 

        self.w = np.random.rand(n_features)
        self.b = 0 

        for _ in range(epcoh):
            for i, x_i in enumerate(x):

                sideOfPlane = y_[i] * (np.dot(x_i,self.w) - self.b) >= 1
                
                if sideOfPlane == True:
                    self.w = self.w - (learningRate * (2*lambda_para*self.w) )
                
                else:
                    self.w = self.w - (learningRate * (2 * lambda_para * self.w - np.dot(x_i,y_[i])))
                    self.b = self.b - (learningRate * y_[i] )

        print(f'weights = {self.w} bias = {self.b}')
        self.plot_svm(x,y)
