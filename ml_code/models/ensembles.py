import numpy as np
import copy
from .trees import DecisionTreeClassifier, DecisionTreeRegressor
from .base import Model

class VotingClassifier(Model):
    """
    Hard-voting ensemble of classification models.
    Each model votes for a class; the majority class is chosen.

    :param models: list or tuple of classification Model instances
    """

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
        """
        Trains all base models on the same data.

        :param x: input features
        :param y: target labels
        :return: None
        """
        for m in self.models:
            m.train(x, y)

        self.trained = True

    def __call__(self, x):
        """
        Predicts class labels by majority vote.

        :param x: input features
        :return: predicted class labels
        """
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
    """
    Averaging ensemble of regression models.
    Final prediction is the mean of all base model predictions.

    :param models: list or tuple of regression Model instances
    """

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
        """
        Trains all base models on the same data.

        :param x: input features
        :param y: target values
        :return: None
        """
        for m in self.models:
            m.train(x, y)

        self.trained = True

    def __call__(self, x):
        """
        Predicts values by averaging base model outputs.

        :param x: input features
        :return: predicted values
        """
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
    """
    Bootstrap aggregating (bagging) ensemble for classification.
    Trains multiple copies of a base model on bootstrap samples and aggregates by majority vote.

    :param model: base classification Model instance to be cloned and trained
    :param n_models: number of base models in the ensemble
    :param samples: fraction (0-1] or absolute number of samples for each bootstrap draw
    :param seed: random seed for reproducibility
    """

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
        """
        Draws a bootstrap sample from the data.

        :param x: input features
        :param y: target labels
        :param rng: numpy random generator
        :return: bootstrapped features and labels
        """
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
        """
        Trains the bagging ensemble on bootstrap samples.

        :param x: input features
        :param y: target labels
        :return: None
        """
        self._trained_models = []
        rng = np.random.default_rng(self.seed)

        for _ in range(self.n_models):
            x_b, y_b = self._bootstrap_samples(x, y, rng)
            m = copy.deepcopy(self.model)
            m.train(x_b, y_b)
            self._trained_models.append(m)

        self.trained = True

    def __call__(self, x):
        """
        Predicts class labels by majority vote over the ensemble.

        :param x: input features
        :return: predicted class labels
        """
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
    """
    Bootstrap aggregating (bagging) ensemble for regression.
    Trains multiple copies of a base model on bootstrap samples and averages predictions.

    :param model: base regression Model instance to be cloned and trained
    :param n_models: number of base models in the ensemble
    :param samples: fraction (0-1] or absolute number of samples for each bootstrap draw
    :param seed: random seed for reproducibility
    """

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
        """
        Draws a bootstrap sample from the data.

        :param x: input features
        :param y: target values
        :param rng: numpy random generator
        :return: bootstrapped features and targets
        """
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
        """
        Trains the bagging ensemble on bootstrap samples.

        :param x: input features
        :param y: target values
        :return: None
        """
        self._trained_models = []
        rng = np.random.default_rng(self.seed)

        for _ in range(self.n_models):
            x_b, y_b = self._bootstrap_samples(x, y, rng)
            m = copy.deepcopy(self.model)
            m.train(x_b, y_b)
            self._trained_models.append(m)

        self.trained = True

    def __call__(self, x):
        """
        Predicts values by averaging the ensemble outputs.

        :param x: input features
        :return: predicted values
        """
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
    """
    Random Forest classifier: bagging of decision trees with feature subsampling.

    :param n_models: number of trees in the forest
    :param max_depth: maximum depth of each tree
    :param seed: random seed for reproducibility
    """

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
    """
    Random Forest regressor: bagging of decision trees with feature subsampling.

    :param n_models: number of trees in the forest
    :param max_depth: maximum depth of each tree
    :param seed: random seed for reproducibility
    """

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
    """
    Extremely Randomized Trees classifier.
    Like Random Forest, but thresholds are chosen randomly.

    :param n_models: number of trees in the ensemble
    :param max_depth: maximum depth of each tree
    :param seed: random seed for reproducibility
    """

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
    """
    Extremely Randomized Trees regressor.
    Like Random Forest, but thresholds are chosen randomly.

    :param n_models: number of trees in the ensemble
    :param max_depth: maximum depth of each tree
    :param seed: random seed for reproducibility
    """

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
