import numpy as np

def make_regression(function='linear', n_samples=100, n_features=10, noise_coef=0, seed=None):
    """
    Creates a custom synthetic dataset for regression tasks. Can be used to test regression models

    :param function: type of dataset: ('linear', 'sin', 'polynomial', 'mixed')
    :param n_samples: number of samples in the dataset
    :param n_features: number of features
    :param noise_coef: controllable parameter of 'mess' in the data
    :param seed: random seed for reproducibility
    :return: data, labels
    """
    supported = ('linear', 'sin', 'polynomial', 'mixed')

    if function not in supported:
        raise ValueError(f'unknown type: {function}; supported types: {supported}')

    rng = np.random.default_rng(seed)

    x = rng.normal(size=(n_samples, n_features))
    w = rng.uniform(2, 10, size=n_features) # the boundaries are arbitrary tbh
    b = rng.uniform(2, 10)

    noise = rng.normal(size=n_samples) * noise_coef

    z = x @ w + b

    if function == 'linear':
        y = z
    elif function == 'polynomial':
        y = 0.1 * z ** 4 - 0.5 * z ** 3 + z ** 2 - 2 * z
    elif function == 'sin':
        y = np.sin(z)
    else:
        y = np.sin(z) + 0.5 * np.sin(5 * z) + 0.2 * np.cos(11 * z)

    y += noise

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