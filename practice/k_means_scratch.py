import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


class KMeansClustering:
    def __init__(self, X, num_clusters):
        self.K = num_clusters
        self.num_examples = X.shape[0]
        self.num_features = X.shape[1]
        self.max_iterations = 100
        self.plot_figures = True
        self.centroids = np.zeros((self.K, self.num_features))

    def initialize_random_centroids(self, X):
        for k in range(self.K):
            centroid = X[np.random.choice(range(self.num_examples), replace=False)]
            self.centroids[k] = centroid
        return self.centroids

    def create_clusters(self, X, centroids):
        clusters = [[] for _ in range(self.K)]

        for point_idx, point in enumerate(X):
            closest_point = np.argmin(
                np.sqrt(np.sum((point - centroids) ** 2, axis=1))
            )
            clusters[closest_point].append(point_idx)
        return clusters

    def calculate_new_centroids(self, X, clusters):
        centroids = np.zeros((self.K, self.num_features))
        for idx, cluster in enumerate(clusters):
            new_centroid = np.mean(X[cluster], axis=0)
            centroids[idx] = new_centroid

        return centroids

    def predict_clusters(self, X):
        y_pred = np.zeros(X.shape[0])

        for idx, point in enumerate(X):
            diff = np.sqrt(np.sum((point - self.centroids) ** 2, axis=1))
            closest_centroid = np.argmin(diff)
            y_pred[idx] = closest_centroid

        return y_pred

        # for cluster_idx, cluster in enumerate(clusters):
        #     for point in cluster:
        #         y_pred[point] = cluster_idx
        #
        # return y_pred

    def plot_fig(self, X, y):
        plt.scatter(X[:, 0], X[:, 1], c=y, s=2)
        plt.show()

    def fit(self, X):
        self.centroids = self.initialize_random_centroids(X)

        for it in range(self.max_iterations):
            clusters = self.create_clusters(X, self.centroids)

            previous_centroids = self.centroids
            self.centroids = self.calculate_new_centroids(X, clusters)

            diff = self.centroids - previous_centroids
            if not diff.any():
                print("Termination criterion satisfied")
                break
        y_pred = self.predict_clusters(X)
        if self.plot_figures:
            self.plot_fig(X, y_pred)

        return y_pred, self.centroids


def main():
    np.random.seed(1)
    num_clusters = 3
    X, _ = make_blobs(n_samples=2000, n_features=2, centers=num_clusters)
    np.random.shuffle(X)
    X_train = X[:1900]
    X_test = X[1900:]
    kmeans = KMeansClustering(X_train, num_clusters)
    kmeans.fit(X_train)
    y = kmeans.predict_clusters(X_test)
    kmeans.plot_fig(X_test, y)


if __name__ == '__main__':
    main()
