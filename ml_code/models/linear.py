import numpy as np
from .base import Model

class LinearRegression(Model):
    """
    Linear regression model trained with gradient descent.
    Supports L1 (Lasso) and L2 (Ridge) regularization and early stopping.

    :param max_iter: maximum number of gradient descent iterations
    :param lr: learning rate
    :param tol: minimum loss improvement to reset early stopping counter
    :param early_stopping_patience: number of iterations without improvement before stopping
    :param ridge_coef: L2 regularization coefficient
    :param lasso_coef: L1 regularization coefficient
    """

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
        """
        Trains the linear regression model using gradient descent.

        :param x: input features of shape (n_samples, n_features)
        :param y: target values of shape (n_samples,)
        :return: list of loss values for each iteration
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)

        self._w = np.zeros(x.shape[1])
        self._b = 0
        losses = []
        steps_no_improvement = 0

        for _ in range(self.max_iter):
            # forward
            y_pred = x @ self._w + self._b # it was just self(x), but now it triggers RuntimeError on __call__
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
        """
        Makes predictions with the trained model.

        :param x: input features of shape (n_samples, n_features) or (n_features,)
        :return: predicted values
        """
        if not self.trained:
            raise RuntimeError(f'{self.__class__.__name__} is not trained')

        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)

        return x @ self._w + self._b

class LogisticRegression(Model):
    """
    Binary logistic regression model trained with gradient descent.
    Uses sigmoid activation and binary cross-entropy loss.

    :param max_iter: maximum number of gradient descent iterations
    :param lr: learning rate
    :param tol: minimum loss improvement to reset early stopping counter
    :param early_stopping_patience: number of iterations without improvement before stopping
    :param threshold: decision threshold for converting probabilities to class labels
    """

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
        """
        Trains the logistic regression model using gradient descent.

        :param x: input features of shape (n_samples, n_features)
        :param y: binary target labels of shape (n_samples,)
        :return: list of loss values for each iteration
        """
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
        """
        Makes class predictions with the trained model.

        :param x: input features of shape (n_samples, n_features) or (n_features,)
        :return: predicted class labels (0 or 1)
        """
        if not self.trained:
            raise RuntimeError(f'{self.__class__.__name__} is not trained')

        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)

        probs = self._sigmoid(x @ self._w + self._b)
        return (probs >= self.threshold).astype(np.int32)