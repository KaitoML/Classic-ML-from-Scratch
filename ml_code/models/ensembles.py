import numpy as np
import copy
from .trees import DecisionTreeClassifier, DecisionTreeRegressor
from .base import Model

class VotingClassifier(Model):
    def __init__(self, models):
        super().__init__(task='classification')

        if not isinstance(models, (list, tuple)):
            raise TypeError('models must be in format of list or tuple')

        for m in models:
            if not isinstance(m, Model):
                raise TypeError('only instances of class Model are supported')

            if m.task != 'classification':
                raise ValueError('only classification models are supported')

        self.models = list(models)

    def __repr__(self):
        return (f'VotingClassifier('
                f'\n    models={[m.__class__.__name__ for m in self.models]},'
                f'\n    trained={self.trained}'
                f'\n)')

    def train(self, x, y):
        for m in self.models:
            m.train(x, y)

        self.trained = True

    def __call__(self, x):
        if not self.trained:
            raise RuntimeError(f'{self.__class__.__name__} is not trained')

        preds = []

        for m in self.models:
            pred = m(x)
            preds.append(pred)

        preds = np.vstack(preds)

        maj_voting_preds = []

        for col in preds.T:
            values, counts = np.unique(col, return_counts=True)
            most_freq = values[counts.argmax()]
            maj_voting_preds.append(most_freq)

        return np.array(maj_voting_preds)

class VotingRegressor(Model):
    def __init__(self, models):
        super().__init__(task='regression')

        if not isinstance(models, (list, tuple)):
            raise TypeError('models must be in format of list or tuple')

        for m in models:
            if not isinstance(m, Model):
                raise TypeError('only instances of class Model are supported')

            if m.task != 'regression':
                raise ValueError('only regression models are supported')

        self.models = list(models)

    def __repr__(self):
        return (f'VotingRegressor('
                f'\n    models={[m.__class__.__name__ for m in self.models]},'
                f'\n    trained={self.trained}'
                f'\n)')

    def train(self, x, y):
        for m in self.models:
            m.train(x, y)

        self.trained = True

    def __call__(self, x):
        if not self.trained:
            raise RuntimeError(f'{self.__class__.__name__} is not trained')

        preds = []

        for m in self.models:
            pred = m(x)
            preds.append(pred)

        preds = np.vstack(preds)

        out = []

        for col in preds.T:
            mean_pred = np.mean(col)
            out.append(mean_pred)

        return np.array(out)

class BaggingClassifier(Model):
    def __init__(self, model, n_models=10, samples=1.0, seed=None):
        super().__init__(task='classification')

        if not isinstance(model, Model):
            raise ValueError('only instances of class Model are supported')

        if model.task != 'classification':
            raise ValueError('only classification models are supported')

        if not isinstance(samples, (float, int)):
            raise ValueError('samples must be either float between 0 and 1, or integer value')

        self.seed = seed
        self.model = model
        self.n_models = n_models
        self.samples = samples
        self._trained_models = []

    def __repr__(self):
        return (f'BaggingClassifier('
                f'\n    model={self.model.__class__.__name__},'
                f'\n    n_models={self.n_models},'
                f'\n    samples={self.samples},'
                f'\n    trained={self.trained},'
                f'\n    seed={self.seed}'
                f'\n)')

    def _bootstrap_samples(self, x, y, rng):
        if 0 < self.samples <= 1:
            n_samples = int(len(y) * self.samples)
        elif 1 < self.samples <= len(y):
            n_samples = self.samples
        else:
            raise IndexError('invalid number of samples')

        indices = rng.integers(0, len(y), n_samples)

        x_b = x[indices]
        y_b = y[indices]
        return x_b, y_b

    def train(self, x, y):
        self._trained_models = []
        rng = np.random.default_rng(self.seed)

        for _ in range(self.n_models):
            x_b, y_b = self._bootstrap_samples(x, y, rng)
            m = copy.deepcopy(self.model)
            m.train(x_b, y_b)
            self._trained_models.append(m)

        self.trained = True

    def __call__(self, x):
        if not self.trained:
            raise RuntimeError(f'{self.__class__.__name__} is not trained')

        preds = []

        for m in self._trained_models:
            pred = m(x)
            preds.append(pred)

        preds = np.vstack(preds)

        maj_voting_preds = []

        for col in preds.T:
            values, counts = np.unique(col, return_counts=True)
            most_freq = values[counts.argmax()]
            maj_voting_preds.append(most_freq)

        return np.array(maj_voting_preds)

class BaggingRegressor(Model):
    def __init__(self, model, n_models=10, samples=1.0, seed=None):
        super().__init__(task='regression')

        if not isinstance(model, Model):
            raise ValueError('only instances of class Model are supported')

        if model.task != 'regression':
            raise ValueError('only regression models are supported')

        if not isinstance(samples, (float, int)):
            raise ValueError('samples must be either float between 0 and 1, or integer value')

        self.seed = seed
        self.model = model
        self.n_models = n_models
        self.samples = samples
        self._trained_models = []

    def __repr__(self):
        return (f'BaggingRegressor('
                f'\n    model={self.model.__class__.__name__},'
                f'\n    n_models={self.n_models},'
                f'\n    samples={self.samples},'
                f'\n    trained={self.trained},'
                f'\n    seed={self.seed}'
                f'\n)')

    def _bootstrap_samples(self, x, y, rng):
        if 0 < self.samples <= 1:
            n_samples = int(len(y) * self.samples)
        elif 1 < self.samples <= len(y):
            n_samples = self.samples
        else:
            raise IndexError('invalid number of samples')

        indices = rng.integers(0, len(y), n_samples)

        x_b = x[indices]
        y_b = y[indices]
        return x_b, y_b

    def train(self, x, y):
        self._trained_models = []
        rng = np.random.default_rng(self.seed)

        for _ in range(self.n_models):
            x_b, y_b = self._bootstrap_samples(x, y, rng)
            m = copy.deepcopy(self.model)
            m.train(x_b, y_b)
            self._trained_models.append(m)

        self.trained = True

    def __call__(self, x):
        if not self.trained:
            raise RuntimeError(f'{self.__class__.__name__} is not trained')

        preds = []

        for m in self._trained_models:
            pred = m(x)
            preds.append(pred)

        preds = np.vstack(preds)

        out = []

        for col in preds.T:
            mean_pred = np.mean(col)
            out.append(mean_pred)

        return np.array(out)

class RandomForestClassifier(BaggingClassifier):
    def __init__(self, n_models=50, max_depth=5, seed=None):
        super().__init__(model=DecisionTreeClassifier(max_depth=max_depth, max_features='sqrt'),
                         n_models=n_models,
                         samples=0.75, seed=seed)
        self.max_depth = max_depth

    def __repr__(self):
        return (f'RandomForestClassifier('
                f'\n    n_models={self.n_models},'
                f'\n    max_depth={self.max_depth},'
                f'\n    trained={self.trained},'
                f'\n    seed={self.seed}'
                f'\n)')

class RandomForestRegressor(BaggingRegressor):
    def __init__(self, n_models=50, max_depth=5, seed=None):
        super().__init__(model=DecisionTreeRegressor(max_depth=max_depth, max_features='sqrt'),
                         n_models=n_models,
                         samples=0.75, seed=seed)
        self.max_depth = max_depth

    def __repr__(self):
        return (f'RandomForestRegressor('
                f'\n    n_models={self.n_models},'
                f'\n    max_depth={self.max_depth},'
                f'\n    trained={self.trained},'
                f'\n    seed={self.seed}'
                f'\n)')

class ExtraTreesClassifier(BaggingClassifier):
    def __init__(self, n_models=50, max_depth=5, seed=None):
        super().__init__(model=DecisionTreeClassifier(max_depth=max_depth, max_features='sqrt', random_thresholds=True),
                         n_models=n_models,
                         samples=0.75, seed=seed)
        self.max_depth = max_depth

    def __repr__(self):
        return (f'ExtraTreesClassifier('
                f'\n    n_models={self.n_models},'
                f'\n    max_depth={self.max_depth},'
                f'\n    trained={self.trained},'
                f'\n    seed={self.seed}'
                f'\n)')

class ExtraTreesRegressor(BaggingRegressor):
    def __init__(self, n_models=50, max_depth=5, seed=None):
        super().__init__(model=DecisionTreeRegressor(max_depth=max_depth, max_features='sqrt', random_thresholds=True),
                         n_models=n_models,
                         samples=0.75, seed=seed)
        self.max_depth = max_depth

    def __repr__(self):
        return (f'ExtraTreesRegressor('
                f'\n    n_models={self.n_models},'
                f'\n    max_depth={self.max_depth},'
                f'\n    trained={self.trained},'
                f'\n    seed={self.seed}'
                f'\n)')