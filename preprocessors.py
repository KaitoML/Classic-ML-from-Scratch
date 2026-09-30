import numpy as np
from models import Model

class Preprocessor:
    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def fit(self, data):
        raise NotImplementedError(f'fit method of {self.__class__.__name__} is not defined')

    def transform(self, data):
        raise NotImplementedError(f'transform method of {self.__class__.__name__} is not defined')

    def fit_transform(self, data):
        self.fit(data)
        data = self.transform(data)
        return data

class StandardScaler(Preprocessor):
    def __init__(self):
        self.mu = None
        self.sigma = None

    def fit(self, data):
        self.mu = np.mean(data, axis=0)
        self.sigma = np.std(data, axis=0)

    def transform(self, data):
        return (data - self.mu) / (self.sigma + 1e-12)

class MinMaxScaler(Preprocessor):
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, data):
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)

    def transform(self, data):
        return (data - self.min) / (self.max - self.min + 1e-12)

class PolynomialFeatures(Preprocessor):
    def __init__(self, degree=2):
        self.degree = degree
        self.features_in = None

    def fit(self, data):
        if data.ndim == 1:
            data = np.expand_dims(data, axis=1)

        self.features_in = data.shape[1]

    def transform(self, data):
        if data.ndim == 1:
            data = np.expand_dims(data, axis=1)

        assert self.features_in == data.shape[1], f'Dimensionality mismatch: {self.features_in} features on fit, {data.shape[1]} features on transform'

        features = [data]
        for d in range(2, self.degree + 1):
            features.append(data ** d)

        data_poly = np.hstack(features)
        return data_poly

# ROUGH IDEA:
class Pipeline:
    def __init__(self, components):
        for c in components:
            assert isinstance(c, (Preprocessor, Model)), f'{c} <-- is not compatible with Pipeline class'

        self.components = components

    def __repr__(self):
        return f'Pipeline({self.components})'

    def train(self, data):

        for c in self.components:
            if isinstance(c, Preprocessor):
                data = c.fit_transform(data)
            else:
                losses = c.train(data)

    def __call__(self, data):
        for c in self.components:
            if isinstance(c, Preprocessor):
                data = c.transform(data)
            else:
                y_pred = c(data)