import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

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