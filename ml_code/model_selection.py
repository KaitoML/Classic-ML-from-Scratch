import numpy as np
import copy
from itertools import product

def random_split(x, y, test_size=0.2, seed=42):
    """
    Performs a single random split into two sets of data: training and testing/validation

    :param x: input data
    :param y: input labels
    :param test_size: the proportion of data that should go to the testing set
    :param seed: random seed for reproducibility
    :return: x_train, x_test, y_train, y_test
    """
    assert isinstance(test_size, float), 'Test size must be float between 0 and 1'

    np.random.seed(seed)
    p = np.random.permutation(len(x))
    x_p, y_p = x[p], y[p]

    n = int(test_size * len(y))
    assert n > 0, f'Invalid test size. Could not calculate {test_size*100:.0f}% of it as a whole number'

    x_train = x_p[n:]
    x_test = x_p[:n]
    y_train = y_p[n:]
    y_test = y_p[:n]

    return x_train, x_test, y_train, y_test

def cv_score(x, y, model, metric, n_folds=5, return_value='all', seed=None):
    """
    Calculates cross validation scores given data, model and target metric

    :param x: input data
    :param y: input labels
    :param model that is being tested
    :param metric: target metric that estimates the quality of the model
    :param n_folds: number of splits made for testing
    :param return_value: defines whether to return all scores or only the average; can be set to either 'all' or 'mean'
    :param seed: random seed for reproducibility
    :return: scores or their average
    """
    assert return_value in ('all', 'mean'), 'Invalid return value. Can be set to "all" or "mean" only'
    assert n_folds <= len(y), 'Invalid number of folds: n_folds > num samples'

    if seed is not None:
        np.random.seed(seed)

    p = np.random.permutation(len(x))
    x_p, y_p = x[p], y[p]

    scores = []
    samples_per_fold = len(y) // n_folds

    x_folds = []
    y_folds = []

    for i in range(n_folds):
        x_fold = x_p[i*samples_per_fold : (i+1)*samples_per_fold]
        y_fold = y_p[i*samples_per_fold : (i+1)*samples_per_fold]

        x_folds.append(x_fold)
        y_folds.append(y_fold)

    for i in range(n_folds):
        x_train = np.concatenate([x_folds[j] for j in range(n_folds) if j != i])
        y_train = np.concatenate([y_folds[j] for j in range(n_folds) if j != i])
        x_test = x_folds[i]
        y_test = y_folds[i]

        _ = model.train(x_train, y_train)
        y_pred = model(x_test)
        score = metric(y_test, y_pred)
        scores.append(float(score))

    return scores if return_value == 'all' else np.mean(scores)

class GridSearchCV:
    def __init__(self, param_grid, model, metric, folds=3, maximize_metric=True, seed=None):
        self.param_grid = param_grid
        self.model = model
        self.metric = metric
        self.folds = folds
        self.maximize_metric = maximize_metric # defines whether greater is better for chosen metric
        self.seed = seed
        self.best_model = None

    def __repr__(self):
        return (f'GridSearch('
                f'\n    model={self.model.__class__.__name__}'
                f'\n)')

    def train(self, x, y):

        combinations = [
            dict(zip(self.param_grid.keys(), values))
            for values in product(*self.param_grid.values())
        ]

        model_to_score = [] # [(model, score)]

        for i in range(len(combinations)):
            c = combinations[i]

            model = copy.deepcopy(self.model)
            for par, val in c.items():
                setattr(model, par, val)

            score = cv_score(x, y, model, self.metric, n_folds=self.folds, return_value='mean', seed=self.seed)
            model_to_score.append((model, score))

        if self.maximize_metric:
            self.best_model = max(model_to_score, key=lambda x: x[1])[0]
        else:
            self.best_model = min(model_to_score, key=lambda x: x[1])[0]


if __name__ == '__main__':
    from utils import make_regression
    from models import RandomForestRegressor
    from metrics import r_squared

    x, y = make_regression(n_samples=300, n_features=3, noise_coef=1)
    x_tr, x_te, y_tr, y_te = random_split(x, y)

    params = {
        'n_models': [10, 15, 20],
        'max_depth': [2, 5, 7]
    }
    m = RandomForestRegressor()

    grid = GridSearchCV(params, m, r_squared)
    grid.train(x_tr, y_tr)
    m = grid.best_model
    print(m)

    m.train(x_tr, y_tr)
    y_pred = m(x_te)
    print(r_squared(y_te, y_pred))