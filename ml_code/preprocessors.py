import numpy as np

class Preprocessor:
    """
    Base class for all data preprocessors.
    """

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def fit(self, data):
        """
        Learns parameters from the data.

        :param data: input data
        :return: None
        """
        raise NotImplementedError(f'fit method of {self.__class__.__name__} is not defined')

    def transform(self, data):
        """
        Applies the fitted transformation to the data.

        :param data: input data
        :return: transformed data
        """
        raise NotImplementedError(f'transform method of {self.__class__.__name__} is not defined')

    def fit_transform(self, data):
        """
        Fits the preprocessor and transforms the data in one step.

        :param data: input data
        :return: transformed data
        """
        self.fit(data)
        data = self.transform(data)
        return data

class StandardScaler(Preprocessor):
    """
    Standardizes features by removing the mean and scaling to unit variance.
    """

    def __init__(self):
        self.mu = None
        self.sigma = None

    def fit(self, data):
        """
        Computes mean and standard deviation for each feature.

        :param data: input data of shape (n_samples, n_features)
        :return: None
        """
        if data.ndim == 1:
            data = np.expand_dims(data, axis=1)

        self.mu = np.mean(data, axis=0)
        self.sigma = np.std(data, axis=0)

    def transform(self, data):
        """
        Standardizes the data using previously computed mean and std.

        :param data: input data of shape (n_samples, n_features)
        :return: standardized data
        """
        data = data.copy()

        if data.ndim == 1:
            data = np.expand_dims(data, axis=1)

        return (data - self.mu) / (self.sigma + 1e-12)

class MinMaxScaler(Preprocessor):
    """
    Scales features to a fixed range by using minimum and maximum values.
    """

    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, data):
        """
        Computes minimum and maximum for each feature.

        :param data: input data of shape (n_samples, n_features)
        :return: None
        """
        if data.ndim == 1:
            data = np.expand_dims(data, axis=1)

        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)

    def transform(self, data):
        """
        Scales the data using previously computed min and max.

        :param data: input data of shape (n_samples, n_features)
        :return: scaled data
        """
        data = data.copy()

        if data.ndim == 1:
            data = np.expand_dims(data, axis=1)

        return (data - self.min) / (self.max - self.min + 1e-12)

class SimpleImputer(Preprocessor):
    """
    Fills NaN values in input data with mean/median/zero of the corresponding columns
    """

    def __init__(self, strategy='mean'):
        if strategy not in ('mean', 'median', 'zeros'):
            raise ValueError('unknown imputer strategy; available: ("mean", "median", "zeros"). ')

        self.strategy = strategy
        self.fill_vals = None


    def fit(self, data):
        """
        Computes fill value for each column

        :param data: input data of shape (n_samples, n_features)
        :return: None
        """
        if data.ndim == 1:
            data = np.expand_dims(data, axis=1)

        fill_vals = []
        for col in data.T:
            if self.strategy == 'mean':
                fill_val = np.nanmean(col)
            elif self.strategy == 'median':
                fill_val = np.nanmedian(col)
            else:
                fill_val = 0

            fill_vals.append(fill_val)

        self.fill_vals = fill_vals

    def transform(self, data):
        """
        Fills missing values using previously computed mean/median or zeros.

        :param data: input data of shape (n_samples, n_features)
        :return: processed data
        """
        data = data.copy()

        if data.ndim == 1:
            data = np.expand_dims(data, axis=1)

        for i, val in enumerate(self.fill_vals):
            col = data[:, i]
            col[np.isnan(col)] = val

        return data

class PolynomialFeatures(Preprocessor):
    """
    Generates polynomial features up to a given degree.
    For degree=2 and input [x1, x2] produces [x1, x2, x1^2, x2^2].

    :param degree: maximum polynomial degree
    """

    def __init__(self, degree=2):
        self.degree = degree
        self.features_in = None

    def fit(self, data):
        """
        Records the number of input features.

        :param data: input data of shape (n_samples, n_features)
        :return: None
        """
        if data.ndim == 1:
            data = np.expand_dims(data, axis=1)

        self.features_in = data.shape[1]

    def transform(self, data):
        """
        Expands the data with polynomial features.

        :param data: input data of shape (n_samples, n_features)
        :return: data with polynomial features appended
        """
        data = data.copy()

        if data.ndim == 1:
            data = np.expand_dims(data, axis=1)

        assert self.features_in == data.shape[1], f'Dimensionality mismatch: {self.features_in} features on fit, {data.shape[1]} features on transform'

        features = [data]
        for d in range(2, self.degree + 1):
            features.append(data ** d)

        data_poly = np.hstack(features)
        return data_poly
