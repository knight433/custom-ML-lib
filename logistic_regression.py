import numpy as np
import matplotlib.pyplot as plt
import random as r

class LogisticRegression:

    def __init__(self):
       self.weights = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def deriveSigmoid(self, z):
       sigOfZ = self.sigmoid(z)
       return  sigOfZ*(1-sigOfZ)
    
    def step(self,z):
       
       return 1 if z>0 else 0

    def plotLine(self,bias,coeff,X,y):
       
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
        
        self.plotLine(self.weights[0],self.weights[1:],X,y_labels)
        return bias, weights
    
    def sloveGradientMethod(self,x_par,y_labels,lerning_rate=0.1,epoch=1000):
        
        X = np.insert(x_par,0,1,axis=1)
        self.weights = np.ones(X.shape[1])

        for _ in range(epoch):
        
            y_hat = self.sigmoid(np.dot(X,self.weights))
            self.weights = self.weights + lerning_rate*(np.dot((y_labels-y_hat),X)/X.shape[0])
        
        bias = self.weights[0]
        weights = self.weights[1:]
        
        self.plotLine(self.weights[0],self.weights[1:],x_par,y_labels)
        return bias, weights
        