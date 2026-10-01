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