class Model:
    """
    Base class for all models in the framework.

    :param task: type of the learning task, must be "regression" or "classification"
    """

    def __init__(self, task):
        self.task = task
        if self.task not in ('regression', 'classification', 'clustering'):
            raise ValueError(f'task of {self.__class__.__name__} must be specified: "regression", "classification", "clustering".')

        self.trained = False

    def __repr__(self):
        raise NotImplementedError(f'__repr__ method of {self.__class__.__name__} is not defined')

    def train(self, x, y):
        """
        Trains the model on the given data.

        :param x: input features
        :param y: target labels
        :return: implementation-defined (often loss history or None)
        """
        raise NotImplementedError(f'train method of {self.__class__.__name__} is not defined')

    def __call__(self, x):
        """
        Makes predictions with the trained model.

        :param x: input features
        :return: predictions
        """
        raise NotImplementedError(f'__call__ method of {self.__class__.__name__} is not defined')
