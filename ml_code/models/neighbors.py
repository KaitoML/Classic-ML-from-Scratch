import numpy as np
from collections import Counter
from .base import Model

class KNNClassifier(Model):
    def __init__(self, n_neighbors=5):
        super().__init__(task='classification')
        self.n_neighbors = n_neighbors
        self._data = None
        self._labels = None

    def __repr__(self):
        return (f"KNNClassifier("
                f"\n    n_neighbors={self.n_neighbors},"
                f"\n    trained={self.trained}"
                f"\n)")

    @staticmethod
    def _euclidean_distance(coords1, coords2):
        return np.sqrt(np.sum((coords1 - coords2) ** 2))

    def train(self, x, y):
        assert self.n_neighbors <= len(x), 'n_neighbors > the number of samples'

        self._data = x
        self._labels = y

        self.trained = True

    def __call__(self, x):
        if not self.trained:
            raise RuntimeError(f'{self.__class__.__name__} is not trained')

        y_pred = []

        for x_i in x:
            # calculate distances from x_i to every datapoint, with a corresponding label
            label_dist = sorted(
                [(self._euclidean_distance(x_i, datapoint), label)
                for datapoint, label in zip(self._data, self._labels)],
                key=lambda pair: pair[0]) # tie-breaker for equal distances (so that KNN won't start comparing labels, which might not be digits)

            # take n_closest datapoints with their labels
            n_closest = label_dist[:self.n_neighbors]

            # count how many datapoints of each label occurred
            label_counter = Counter(int(label) for _, label in n_closest)

            most_freq_label = max(label_counter, key=label_counter.get)
            y_pred.append(most_freq_label)

        y_pred = np.array(y_pred)
        return y_pred