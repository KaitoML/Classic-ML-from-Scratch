import numpy as np
from .base import Model

class Centroid:
    def __init__(self, x, boundaries, seed=None):
        self._rng = np.random.default_rng(seed)
        self.data = []
        self.mean_distance = None

        self.coords = np.zeros(x.shape[1])
        for j, val in enumerate(self.coords):
            low, high = boundaries[j]
            self.coords[j] = self._rng.uniform(low, high)

    def __repr__(self):
        return (f'Centroid('
                f'\n    coords={self.coords},'
                f'\n    data={self.data}'
                f'\n)')

    def assign(self, x_i):
        self.data.append(x_i)

    def reset(self):
        self.data = []

    def move(self):
        if len(self.data) == 0:
            return

        self.coords = np.mean(self.data, axis=0)

    def estimate_inertia(self):
        if len(self.data) == 0:
            return np.inf

        data = np.asarray(self.data)
        return np.sum((data - self.coords) ** 2)

class KMeans(Model):
    def __init__(self, n_clusters=5, n_inits=10, max_iter=100, tol=1e-6, seed=None):
        super().__init__(task='clustering')
        self.n_clusters = n_clusters
        self.n_inits = n_inits
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

        self._rng = np.random.default_rng(self.seed)
        self.labels = None
        self._centroids = None

    def __repr__(self):
        return (f'KMeans('
                f'\n    n_clusters={self.n_clusters},'
                f'\n    n_inits={self.n_inits},'
                f'\n    max_iter={self.max_iter},'
                f'\n    tol={self.tol},'
                f'\n    trained={self.trained},'
                f'\n    seed={self.seed}'
                f'\n)')

    @staticmethod
    def _euclidean_distance(coords1, coords2):
        return np.sqrt(np.sum((coords1 - coords2) ** 2))

    def train(self, x, y=None):

        boundaries = {}  # {feature_idx: (min, max)}
        for i, feature in enumerate(x.T):
            boundaries[i] = (np.min(feature), np.max(feature))

        best_centroids = None
        lowest_total_inertia = np.inf
        for init in range(self.n_inits):

            # initialize centroids
            centroids = []
            for i in range(self.n_clusters):
                centroid = Centroid(x, boundaries)
                centroids.append(centroid)

            for _ in range(self.max_iter):
                # reset centroids data
                for c in centroids:
                    c.reset()

                # assign each datapoint to a centroid
                for x_i in x:
                    lowest_dist = np.inf
                    closest_centroid = None

                    for c in centroids:
                        dist = self._euclidean_distance(x_i, c.coords)
                        if dist < lowest_dist:
                            lowest_dist = dist
                            closest_centroid = c

                    closest_centroid.assign(x_i)

                shift = 0
                for c in centroids:
                    old = c.coords.copy()
                    c.move()
                    shift += np.linalg.norm(c.coords - old)

                if shift <= self.tol:
                    break

            total_inertia = 0
            for c in centroids:
                inertia = c.estimate_inertia()
                total_inertia += inertia

            if total_inertia < lowest_total_inertia:
                lowest_total_inertia = total_inertia
                best_centroids = centroids

        self._centroids = best_centroids

        labels = []
        for x_i in x:
            dists = [self._euclidean_distance(c.coords, x_i) for c in self._centroids]
            labels.append(np.argmin(dists))

        self.labels = np.array(labels)
        self.trained = True

    def __call__(self, x):
        if not self.trained:
            raise RuntimeError(f'{self.__class__.__name__} is not trained')

        labels = []
        for x_i in x:
            dists = [self._euclidean_distance(c.coords, x_i) for c in self._centroids]
            labels.append(np.argmin(dists))

        return np.array(labels)


if __name__ == '__main__':
    # TODO:
    #   Add KMeans tests to tests.ipynb
    x = np.array([[1, 5],
                  [1.3, 5.4],
                  [1.1, 6],
                  [3, 2],
                  [4, 3]])

    kmeans = KMeans(n_clusters=2, n_inits=10, max_iter=10)
    kmeans.train(x)
    print(kmeans.labels)