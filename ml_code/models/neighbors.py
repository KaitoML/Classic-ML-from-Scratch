import numpy as np
from collections import Counter
from .base import Model

class KNN(Model):
    """
    Base K-Nearest Neighbors model class.

    :param n_neighbors: number of nearest neighbors to use for prediction
    :param task: 'regression' or 'classification'
    """
    def __init__(self, n_neighbors=5, task=None):
        super().__init__(task=task)
        self.n_neighbors = n_neighbors
        self.task = task

        self._data = None
        self._labels = None

    def __repr__(self):
        raise NotImplementedError(f'__repr__ method of {self.__class__.__name__} is not defined')

    @staticmethod
    def _euclidean_distance(coords1, coords2):
        """
        Calculates Euclidean distance between two points.

        :param coords1: first point
        :param coords2: second point
        :return: Euclidean distance
        """
        return np.sqrt(np.sum((coords1 - coords2) ** 2))

    def train(self, x, y):
        """
        Stores the training data for later nearest-neighbor search.

        :param x: input features of shape (n_samples, n_features)
        :param y: target labels of shape (n_samples,)
        :return: None
        """
        if self.n_neighbors > len(x):
            raise ValueError('n_neighbors > the number of samples')

        self._data = x
        self._labels = y

        self.trained = True

    def __call__(self, x):
        """
        Predicts class labels by majority vote among the k nearest neighbors.

        :param x: input features of shape (n_samples, n_features)
        :return: predicted class labels
        """
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

            if self.task == 'classification':

                # count how many datapoints of each label occurred
                label_counter = Counter(int(label) for _, label in n_closest)

                # find the most common label
                pred = max(label_counter, key=label_counter.get)

            else:
                # calculate mean of the closest datapoints' labels
                pred = np.mean([item[1] for item in n_closest])

            y_pred.append(pred)

        y_pred = np.array(y_pred)
        return y_pred

class KNNClassifier(KNN):
    """
    K-Nearest Neighbors classifier.
    Supports multiclass classification.

    :param n_neighbors: number of nearest neighbors to use for prediction
    """
    def __init__(self, n_neighbors=5):
        super().__init__(task='classification',
                         n_neighbors=n_neighbors)

    def __repr__(self):
        return (f"KNNClassifier("
                f"\n    n_neighbors={self.n_neighbors},"
                f"\n    trained={self.trained}"
                f"\n)")


class KNNRegressor(KNN):
    """
    K-Nearest Neighbors regressor.

    :param n_neighbors: number of nearest neighbors to use for prediction
    """
    def __init__(self, n_neighbors=5):
        super().__init__(task='regression',
                         n_neighbors=n_neighbors)

    def __repr__(self):
        return (f"KNNRegressor("
                f"\n    n_neighbors={self.n_neighbors},"
                f"\n    trained={self.trained}"
                f"\n)")