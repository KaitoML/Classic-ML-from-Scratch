import numpy as np
from .base import Model
from .trees import DecisionTreeClassifier, DecisionTreeRegressor

class AdaBoostClassifier(Model):
    """
    AdaBoost classifier using decision stumps as weak learners.
    Samples are reweighted after each round; final prediction is a weighted vote.

    :param n_models: number of weak learners (stumps)
    :param seed: random seed for reproducibility
    """

    def __init__(self, n_models=50, seed=None):
        super().__init__(task='classification')
        self.n_models = n_models
        self.seed = seed

        self._rng = np.random.default_rng(self.seed)
        self.stumps = []

    def __repr__(self):
        return (f'AdaBoost('
                f'\n   n_models={self.n_models},'
                f'\n   seed={self.seed},'
                f'\n   trained={self.trained}'
                f'\n)')

    def train(self, x, y):
        """
        Trains AdaBoost on the given data.
        Each round samples the data by current weights, fits a stump,
        then updates sample weights using the stump's weighted error.

        :param x: input features of shape (n_samples, n_features)
        :param y: target labels of shape (n_samples,)
        :return: None
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 1)

        n_samples = x.shape[0]
        sample_weights = np.full(x.shape[0], fill_value=1/n_samples)

        for _ in range(self.n_models):

            indices = self._rng.choice(
                n_samples,
                size=n_samples,
                p=sample_weights
            )

            data = x[indices]
            labels = y[indices]

            stump = DecisionTreeClassifier(max_depth=1, seed=self.seed)
            stump.train(data, labels)
            pred = stump(x)
            self.stumps.append(stump)

            # estimate error for a stump
            eps = 1e-12
            total_error = np.sum(sample_weights[pred != y])
            total_error = np.clip(total_error, eps, 1 - eps)

            # estimate amount of say for a stump
            stump.amount_of_say = 0.5 * np.log((1 - total_error)/total_error)

            # update sample weights
            # if error: *= exp(amount of say)
            # if correct: *= exp(-amount of say)
            sample_weights *= np.exp(stump.amount_of_say * (pred != y).astype(float) * 2 - stump.amount_of_say)

            # scale so that they add up to one
            sample_weights /= np.sum(sample_weights)

        self.trained = True

    def __call__(self, x):
        """
        Predicts class labels by weighted majority vote over all stumps.

        :param x: input features of shape (n_samples, n_features)
        :return: predicted class labels
        """
        if not self.trained:
            raise RuntimeError(f'{self.__class__.__name__} is not trained')

        stumps_preds = []

        for s in self.stumps:
            pred = s(x)
            stumps_preds.append(pred)

        stumps_preds = np.array(stumps_preds)

        outs = []

        for options in stumps_preds.T:
            weighted_voting = {} # {class: total say for this class}
            for i, option in enumerate(options):
                weighted_voting[option] = weighted_voting.get(option, 0) + self.stumps[i].amount_of_say

            outs.append(max(weighted_voting, key=weighted_voting.get))

        return np.array(outs)

class GradientBoostRegressor(Model):
    """
    Gradient boosting regressor for squared error loss.
    Sequentially fits regression trees on residuals and combines them with a learning rate.

    :param n_models: number of boosting stages (trees)
    :param max_depth: maximum depth of each regression tree
    :param lr: learning rate (shrinkage) applied to each tree's contribution
    :param seed: random seed for reproducibility
    """

    def __init__(self, n_models=50, max_depth=3, lr=0.1, seed=None):
        super().__init__(task='regression')
        self.n_models = n_models
        self.max_depth = max_depth
        self.lr = lr
        self.seed = seed

        self.base = None
        self.trees = []

    def __repr__(self):
        return (f'GradientBoostRegressor('
                f'\n    n_models={self.n_models},'
                f'\n    max_depth={self.max_depth},'
                f'\n    lr={self.lr},'
                f'\n    seed={self.seed},'
                f'\n    trained={self.trained}'
                f'\n)')

    def train(self, x, y):
        """
        Trains the gradient boosting ensemble.
        Starts from the mean of y, then repeatedly fits trees to the current residuals.

        :param x: input features of shape (n_samples, n_features)
        :param y: target values of shape (n_samples,)
        :return: None
        """
        self.base = np.mean(y)
        previous_pred = np.full(len(y), fill_value=self.base)

        for _ in range(self.n_models):

            resid = y - previous_pred

            tree = DecisionTreeRegressor(max_depth=self.max_depth, seed=self.seed)
            tree.train(x, resid)
            self.trees.append(tree)

            previous_pred += self.lr * tree(x)

        self.trained = True

    def __call__(self, x):
        """
        Makes predictions by summing the base value and all tree contributions.

        :param x: input features of shape (n_samples, n_features)
        :return: predicted values
        """
        if not self.trained:
            raise RuntimeError(f'{self.__class__.__name__} is not trained')

        out = self.base
        for t in self.trees:
            out += self.lr * t(x)

        return out

class GradientBoostClassifier(Model):
    """
    Binary gradient boosting classifier with logistic loss.
    Works in log-odds space: trees are regression trees fitted on probability residuals,
    and class labels are obtained via sigmoid and a 0.5 threshold.

    :param n_models: number of boosting stages (trees)
    :param max_depth: maximum depth of each regression tree
    :param lr: learning rate (shrinkage) applied to each tree's contribution
    :param seed: random seed for reproducibility
    """

    def __init__(self, n_models=50, max_depth=3, lr=0.1, seed=None):
        super().__init__(task='classification')
        self.n_models = n_models
        self.max_depth = max_depth
        self.lr = lr
        self.seed = seed

        self.base = None
        self.trees = []

    def __repr__(self):
        return (f'GradientBoostClassifier('
                f'\n    n_models={self.n_models},'
                f'\n    max_depth={self.max_depth},'
                f'\n    lr={self.lr},'
                f'\n    seed={self.seed},'
                f'\n    trained={self.trained}'
                f'\n)')

    @staticmethod
    def _sigmoid(z):
        """
        Applies the sigmoid function to map log-odds to probabilities.

        :param z: log-odds values
        :return: probabilities in (0, 1)
        """
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def train(self, x, y):
        """
        Trains the binary gradient boosting classifier.
        Starts from log-odds of the class prior, then fits regression trees
        to residuals y - sigmoid(F).

        :param x: input features of shape (n_samples, n_features)
        :param y: binary target labels of shape (n_samples,)
        :return: None
        """
        assert len(np.unique(y)) == 2, "GradientBoostingClassifier only supports binary classification"

        p = np.clip(np.mean(y), 1e-12, 1 - 1e-12)
        self.base = np.log(p/(1-p))

        previous_pred = np.full(len(y), fill_value=self.base)

        for _ in range(self.n_models):
            prob = self._sigmoid(previous_pred)
            resid = y - prob

            tree = DecisionTreeRegressor(max_depth=self.max_depth, seed=self.seed) # regression tree because we need to predict continuous values (resids), not classes
            tree.train(x, resid)
            self.trees.append(tree)

            previous_pred += self.lr * tree(x)

        self.trained = True

    def __call__(self, x):
        """
        Predicts binary class labels.
        Accumulates log-odds, converts them to probabilities with sigmoid,
        and thresholds at 0.5.

        :param x: input features of shape (n_samples, n_features)
        :return: predicted class labels (0 or 1)
        """
        if not self.trained:
            raise RuntimeError(f'{self.__class__.__name__} is not trained')

        out = self.base
        for t in self.trees:
            out += self.lr * t(x)

        out = self._sigmoid(out)
        return (out >= 0.5).astype(int)
