import numpy as np
import matplotlib.pyplot as plt

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

        for count in range(epoch):
            # Calculate predictions
            pred = np.dot(input_ar, weights) + w0  # Dot product of inputs and weights + bias term

            # Compute the loss (Mean Squared Error)
            loss = np.mean((input_labels - pred) ** 2)
            loss_history.append(loss)

            # Calculate gradients
            dL_w0 = -2 * np.sum(input_labels - pred) / n  # Gradient for bias term
            dl_wi = -2 * np.dot((input_labels - pred), input_ar) / n  # Gradient for weights

            # Update weights and bias
            weights -= learningRate * dl_wi
            w0 -= learningRate * dL_w0

                
        self.coef = [w0] + list(weights)  # Combine bias and weights into a single list

        self.plotLR(input_ar,input_labels,self.coef)