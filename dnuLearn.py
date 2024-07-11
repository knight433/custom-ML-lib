import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.animation import FuncAnimation

class LinearRegression:
    
    def __init__(self) -> None:
        self.coef = []

    def getcoef(self):

        return self.coef

    def predict(self,input_par):
        
        w0 = self.coef[0]
        weights = self.coef[1:]

        pred = np.dot(input_par, weights) + w0

        return pred


    def plotLR(self, input_ar, input_labels,coefficients):
        X = np.hstack([np.ones((input_ar.shape[0], 1)), input_ar])
        plot = True
        if plot:
            if input_ar.shape[1] == 1:
                # For 2D plotting
                plt.scatter(input_ar, input_labels, color='blue', label='Data points')
                
                # Generate predictions using the computed coefficients
                predicted_labels = np.dot(X, coefficients)
                
                # Plot the regression line
                plt.plot(input_ar, predicted_labels, color='red', label='Regression line')
                
                plt.xlabel('Feature X')
                plt.ylabel('Target y')
                plt.title('Linear Regression Fit')
                plt.legend()
                plt.show()
            elif input_ar.shape[1] == 2:
                # For 3D plotting
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                
                # Scatter plot of the data points
                ax.scatter(input_ar[:, 0], input_ar[:, 1], input_labels, color='blue', label='Data points')
                
                # Create a mesh grid for the surface plot
                x_surf, y_surf = np.meshgrid(np.linspace(input_ar[:, 0].min(), input_ar[:, 0].max(), 100), 
                                             np.linspace(input_ar[:, 1].min(), input_ar[:, 1].max(), 100))
                z_surf = coefficients[0] + coefficients[1] * x_surf + coefficients[2] * y_surf
                
                # Plot the regression plane
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
        # Add a column of ones to input_ar for the intercept term
        X = np.hstack([np.ones((input_ar.shape[0], 1)), input_ar])
        
        # Compute the coefficients using the normal equation
        X_trans = np.transpose(X)
        part1 = np.dot(X_trans, X)
        part1_inv = np.linalg.inv(part1)
        self.coef = np.dot(np.dot(part1_inv, X_trans), input_labels)

        if plot:
            self.plotLR(input_ar,input_labels,self.coef)

    def gradDestSolve(self, input_ar, input_labels, learningRate=0.1):
        # For loss function 1/n * Σ(yi - ŷi)^2
        # Initialize weights with random values
        weights = np.random.rand(input_ar.shape[1])  # shape[1] to match the number of features
        w0 = np.random.rand()  # Initialize bias term with a random value
        epoch = 100
        n = len(input_labels)
        
        loss_history = []  # For storing loss values over epochs if plotting is needed

        for _ in range(epoch):
            # Calculate predictions
            pred = np.dot(input_ar, weights) + w0  # Dot product of inputs and weights + bias term

            # Compute the loss (Mean Squared Error)
            loss = np.mean((input_labels - pred) ** 2)
            loss_history.append(loss)

            # Calculate gradients
            dL_w0 = -2 * np.sum(input_labels - pred) / n  # Gradient for bias
            dl_wi = -2 * np.dot((input_labels - pred), input_ar) / n  # Gradient for weights

            # Update weights and bias
            weights -= learningRate * dl_wi
            w0 -= learningRate * dL_w0

        self.coef = [w0] + list(weights)  # Combine bias and weights into a single list

        self.plotLR(input_ar,input_labels,self.coef)

class LogisticRegression:

    def __init__(self):
       self.weights = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def step(self,z):
       
       return 1 if z>0 else 0

    def _plotLine(self,bias,coeff,X,y):
       
       if len(coeff) == 2:
        m = -(coeff[0]/coeff[1])
        b = -(bias/coeff[1])

        x_in = np.linspace(-3,3,100)
        y_in = m*x_in + b
        
        plt.plot(x_in,y_in)
        plt.scatter(X[:,0],X[:,1],c=y,cmap='winter',s=100)
        plt.ylim(-3,2)
        plt.show()
    
    def predict(self, X):

        X = np.hstack([1, X])
        predictions = self.sigmoid(np.dot(X, self.weights))
        
        return (predictions >= 0.5).astype(int)

    def solvePerceptron(self, x_par, y_labels, learning_rate=0.1, epoch=1000):
        
        X = x_par
        # Add bias term (column of ones) to the input features
        x_par = np.insert(x_par, 0, 1, axis=1)
        
        # Initialize weights to small random values
        self.weights = np.random.randn(x_par.shape[1])

        for i in range(epoch):
            j = np.random.randint(0, x_par.shape[0])
            y_hat = self.step(np.dot(x_par[j], self.weights)) 
            self.weights = self.weights + learning_rate * (y_labels[j] - y_hat) * x_par[j]
        
        bias = self.weights[0]
        weights = self.weights[1:]
        
        self._plotLine(self.weights[0],self.weights[1:],X,y_labels)
        return bias, weights
    
    def sloveGradientMethod(self,x_par,y_labels,lerning_rate=0.1,epoch=1000):
        
        X = np.insert(x_par,0,1,axis=1)
        self.weights = np.ones(X.shape[1])

        for _ in range(epoch):
        
            y_hat = self.sigmoid(np.dot(X,self.weights))
            self.weights = self.weights + lerning_rate*(np.dot((y_labels-y_hat),X)/X.shape[0])
        
        bias = self.weights[0]
        weights = self.weights[1:]
        
        self._plotLine(self.weights[0],self.weights[1:],x_par,y_labels)
        return bias, weights
    
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

    def train(self, data, k):
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        self.centers = np.random.uniform(low=min_vals, high=max_vals, size=(k, data.shape[1]))
        
        for e in range(self.epoch):
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

            # Save the centers and labels at each step for animation
            self.history.append((self.centers.copy(), cluster_labels.copy()))

        self.y = cluster_labels
        return self.centers

    def animate_training(self, data, k):
        fig, ax = plt.subplots(figsize=(8, 6))

        def update(frame):
            centers, cluster_labels = self.history[frame]
            ax.clear()
            ax.scatter(data[:, 0], data[:, 1], c=cluster_labels, s=10, cmap='viridis')
            ax.scatter(centers[:, 0], centers[:, 1], c='red', s=300, alpha=0.75, marker='X')
            ax.set_title(f'Training Epoch {frame + 1}')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')

        ani = FuncAnimation(fig, update, frames=len(self.history), interval=200, repeat=False)
        plt.show()
    
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
        colors = plt.cm.get_cmap('tab10', len(unique_labels)) #no clue wat it does

        for i, label in enumerate(unique_labels):
            cluster_points = data[labels == label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, c=[colors(i)], label=f'Cluster {label}')

        plt.scatter(self.centers[:, 0], self.centers[:, 1], s=200, c='red', marker='X', label='Centers')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()