n_samples = 200
n_components = 3

X, y_true = make_blobs(n_samples=n_samples, centers=n_components,cluster_std=0.5, random_state=0)

kmean_model = dnuLearn.KMeanClustering(epoch=10)
center = kmean_model.train(X, n_components)
kmean_model._plot_training_progress(X,interval=1000)

n_samples = 200
n_components = 1