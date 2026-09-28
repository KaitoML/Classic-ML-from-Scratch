import numpy as np
import metrics

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

    x_train = x_p[:n]
    x_test = x_p[n:]
    y_train = y_p[:n]
    y_test = y_p[n:]

    return x_train, x_test, y_train, y_test

def cv_score(x, y, model, metric, n_folds=5, return_value='all'):
    """
    Calculates cross validation scores given data, model and target metric

    :param x: input data
    :param y: input labels
    :param model: trained model that is being tested on the data
    :param metric: target metric that estimates the quality of the model
    :param n_folds: number of splits made for testing
    :param return_value: defines whether to return all scores or only the average; can be set to either 'all' or 'mean'
    :return: scores or their average
    """
    assert (return_value == 'all') | (return_value == 'mean'), 'Invalid return value. Can be set to "all" or "mean" only'
    scores = []
    samples_per_fold = len(y) // n_folds

    for i in range(n_folds):
        x_fold = x[i*samples_per_fold : (i+1)*samples_per_fold]
        y_fold = y[i*samples_per_fold : (i+1)*samples_per_fold]

        y_pred = model(x_fold)
        score = metric(y_fold, y_pred)
        scores.append(score)

    return scores if return_value is 'all' else np.mean(scores)

if __name__ == '__main__':
    x = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]])

    y = np.array([10, 20, 30, 40])

    cv_score(x, y, model=None, metric=None, n_folds=4)