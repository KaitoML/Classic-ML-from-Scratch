import numpy as np
from collections import Counter

class Model:
    def __init__(self, task):
        self.task = task
        if self.task not in ('regression', 'classification'):
            raise ValueError(f'task of {self.__class__.__name__} must be specified: "regression", "classification".')

        self.trained = False

    def __repr__(self):
        raise NotImplementedError(f'__repr__ method of {self.__class__.__name__} is not defined')

    def train(self, x, y):
        raise NotImplementedError(f'train method of {self.__class__.__name__} is not defined')

    def __call__(self, x):
        raise NotImplementedError(f'__call__ method of {self.__class__.__name__} is not defined')

# LINEAR MODELS
# ---

class LinearRegression(Model):
    def __init__(self, max_iter=1000, lr=1e-3, tol=1e-6, early_stopping_patience=10, ridge_coef=0.0, lasso_coef=0.0):
        super().__init__(task='regression')
        self.max_iter = max_iter
        self.ridge_coef = ridge_coef
        self.lasso_coef = lasso_coef
        self.lr = lr
        self.tol = tol
        self.early_stopping_patience = early_stopping_patience
        self._w = None
        self._b = None

    def __repr__(self):
        return (f'LinearRegression('
                f'\n    max_iter={self.max_iter},'
                f'\n    lr={self.lr},'
                f'\n    tol={self.tol},'
                f'\n    early_stopping_patience={self.early_stopping_patience},'
                f'\n    ridge_coef={self.ridge_coef},'
                f'\n    lasso_coef={self.lasso_coef},'
                f'\n    trained={self.trained}'
                f'\n)')

    def train(self, x, y):
        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)

        self._w = np.zeros(x.shape[1])
        self._b = 0
        losses = []
        steps_no_improvement = 0

        for _ in range(self.max_iter):
            # forward
            y_pred = self(x)
            errors = y_pred - y
            loss = 1/len(y) * np.sum(errors ** 2)
            loss += self.ridge_coef * np.sum(np.square(self._w))
            loss += self.lasso_coef * np.sum(np.abs(self._w))

            # early stopping
            if len(losses) > 0 and losses[-1] - loss <= self.tol:
                steps_no_improvement += 1
            else:
                steps_no_improvement = 0

            if steps_no_improvement == self.early_stopping_patience:
                break

            losses.append(loss)

            # backward
            dw = 2/len(y) * x.T @ errors
            dw += 2 * self.ridge_coef * self._w
            dw += self.lasso_coef * np.sign(self._w) # np.sign() returns -1 if x < 0, 0 if x = 0, 1 if x > 0
            db = 2/len(y) * np.sum(errors)

            # step
            self._w -= self.lr * dw
            self._b -= self.lr * db

        self.trained = True
        return losses

    def __call__(self, x):
        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)

        return x @ self._w + self._b

class LogisticRegression(Model):
    def __init__(self, max_iter=1000, lr=1e-3, tol=1e-6, early_stopping_patience=10, threshold=0.5):
        super().__init__(task='classification')
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.early_stopping_patience = early_stopping_patience
        self.threshold = threshold
        self._w = None
        self._b = None

    def __repr__(self):
        return (f'LogisticRegression('
                f'\n    max_iter={self.max_iter},'
                f'\n    lr={self.lr},'
                f'\n    tol={self.tol},'
                f'\n    early_stopping_patience={self.early_stopping_patience},'
                f'\n    threshold={self.threshold},'
                f'\n    trained={self.trained}'
                f'\n)')

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def train(self, x, y):
        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)

        self._w = np.zeros(x.shape[1])
        self._b = 0
        losses = []
        steps_no_improvement = 0

        for _ in range(self.max_iter):
            # forward
            z = x @ self._w + self._b
            y_pred = self._sigmoid(z)
            eps = 1e-12
            y_pred = np.clip(y_pred, eps, 1 - eps)
            errors = -y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)
            loss = np.mean(errors)

            # early stopping
            if len(losses) > 0 and losses[-1] - loss <= self.tol:
                steps_no_improvement += 1
            else:
                steps_no_improvement = 0

            if steps_no_improvement == self.early_stopping_patience:
                break

            losses.append(loss)

            # backward
            dw = 1/len(errors) * x.T @ (y_pred - y)
            db = 1/len(errors) * np.sum(y_pred - y)

            # step
            self._w -= self.lr * dw
            self._b -= self.lr * db

        self.trained = True
        return losses

    def __call__(self, x):
        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)

        probs = self._sigmoid(x @ self._w + self._b)
        return (probs >= self.threshold).astype(np.int32)

# NEIGHBORS
# ---

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

# TREES
# ---
# soon

# ENSEMBLES
# ---

class VotingClassifier(Model):
    def __init__(self, models):
        super().__init__(task='classification')

        if not isinstance(models, (list, tuple)):
            raise TypeError('models must be in format of list or tuple')

        for m in models:
            if not isinstance(m, Model):
                raise TypeError('only instances of class Models are supported')

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
                raise TypeError('only instances of class Models are supported')

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