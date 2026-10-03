import numpy as np
from math import pi
from .base import Model

class GaussianNaiveBayes(Model):
    """
    Gaussian Naive Bayes classifier.
    Assumes features are independent and normally distributed within each class.
    """

    def __init__(self):
        super().__init__(task='classification')
        self.priors = None
        self.means = None
        self.vars = None

    def __repr__(self):
        return (f'GaussianNaiveBayes('
                f'\n    trained={self.trained}'
                f'\n)')

    def train(self, x, y):
        """
        Estimates class priors, means and variances for each feature.

        :param x: input features of shape (n_samples, n_features)
        :param y: target labels of shape (n_samples,)
        :return: None
        """

        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)

        self.priors = {}
        self.means = {}
        self.vars = {}

        self.classes = np.unique(y)

        for c in self.classes:
            x_c = x[y == c]
            self.priors[c] =  len(x_c) / len(y)
            self.means[c] = np.mean(x_c, axis=0)
            self.vars[c] = np.var(x_c, axis=0) + 1e-12

        self.trained = True

    @staticmethod
    def _gaussian_logprob(x, mean, var):
        """
        Log-probability of x under a Gaussian with given mean and variance.
        """
        return -0.5 * np.log(2 * pi * var) - ((x - mean) ** 2) / (2 * var)

    def __call__(self, x):
        """
        Predicts class labels.

        :param x: input features of shape (n_samples, n_features) or (n_features,)
        :return: predicted class labels
        """
        if not self.trained:
            raise RuntimeError(f'{self.__class__.__name__} is not trained')

        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)

        outs = []
        for x_i in x:

            best_class = None
            highest_logprob = -np.inf

            for c in self.classes:
                logprob = np.log(self.priors[c])
                logprob += np.sum(self._gaussian_logprob(x_i, self.means[c], self.vars[c]))

                if logprob > highest_logprob:
                    highest_logprob = logprob
                    best_class = c

            outs.append(best_class)

        return np.array(outs)