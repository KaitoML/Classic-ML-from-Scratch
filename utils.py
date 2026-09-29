import numpy as np

def make_regression(n_samples=100, n_features=10, noise_coef=0, seed=None):
    """
    Creates a custom synthetic dataset for regression tasks. Can be used to test regression models

    :param n_samples: number of samples in the dataset
    :param n_features: number of features
    :param noise_coef: controllable parameter of 'mess' in the data
    :param seed: random seed for reproducibility
    :return: data, labels
    """
    rng = np.random.default_rng(seed)

    x = rng.normal(size=(n_samples, n_features))
    w = rng.uniform(2, 10, size=n_features) # the boundaries are arbitrary tbh
    b = rng.uniform(2, 10)

    noise = rng.normal(size=n_samples) * noise_coef

    y = x @ w + b + noise

    return x, y

def make_classification(n_samples=100, n_features=10, n_classes=2, noise_coef=0, seed=None):
    """
    Creates a custom synthetic dataset for binary/multiclass classification tasks. Can be used to test classification models

    :param n_samples: number of samples in the dataset
    :param n_features: number of features
    :param n_classes: number of classes
    :param noise_coef: controllable parameter of 'mess' in the data
    :param seed: random seed for reproducibility
    :return: data, labels
    """
    rng = np.random.default_rng(seed)

    x = rng.normal(size=(n_samples, n_features))
    w = rng.uniform(2, 10, size=(n_features, n_classes))
    b = rng.uniform(2, 10, size=n_classes)

    noise = rng.normal(size=(n_samples, n_classes)) * noise_coef

    y = x @ w + b + noise
    y = np.argmax(y, axis=1)

    return x, y