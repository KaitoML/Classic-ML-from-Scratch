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

if __name__ == '__main__':
    x = np.array([[1, 20, 222, 13],
                  [134, 123, 1, -23]])

    scaler = StandardScaler()
    print(scaler)
    x = scaler.fit_transform(x)
    print(x)

    x = np.array([[1, 20, 222, 13],
                  [134, 123, 1, -23]])

    scaler = MinMaxScaler()
    print(scaler)
    x = scaler.fit_transform(x)
    print(x)