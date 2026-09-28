import numpy as np

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
    :param model: model that is being tested on the data
    :param metric: target metric that estimates the quality of the model
    :param n_folds: number of splits made for testing
    :param return_value: defines whether to return all scores or only the average; can be set to either 'all' or 'mean'
    :param seed: random seed for reproducibility
    :return: scores or their average
    """
    assert return_value in ('all', 'mean'), 'Invalid return value. Can be set to "all" or "mean" only'

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

if __name__ == '__main__':
    x = np.array([[1, 1, 1],
                  [1, 0, 1],
                  [0, 0, 1]])

    y = np.array([1, 1, 0])

    from models import LogisticRegression
    from metrics import accuracy
    model = LogisticRegression()
    scores = cv_score(x, y, model=model, metric=accuracy, n_folds=3, return_value='all')
    print(scores)