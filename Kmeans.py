import numpy as np
import matplotlib.pyplot as plt

class KMeanClustering:

    def __init__(self,epoch=100):
        
        self.epoch = epoch
        self.centers = None

    def _distanceCal(self,x1,x2):
        
        return np.sqrt(np.sum((x1-x2)**2))

    def train(self, data, k):
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        self.centers = np.random.uniform(low=min_vals, high=max_vals, size=(k, data.shape[1]))
        
        for _ in range(self.epoch):
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
                else: # If a cluster loses all points, reinitialize its center randomly
                    new_centers[i] = np.random.uniform(low=min_vals, high=max_vals, size=data.shape[1])
    
            self.centers = new_centers
            
        self.y = cluster_labels
    
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