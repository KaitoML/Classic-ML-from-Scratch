import os
import pickle

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

    def save(self, filepath):
        """
        Saves the trained model to a file using pickle.

        :param filepath: path where the model will be saved (e.g., 'model.pkl')
        :return: None
        """
        if not self.trained:
            raise RuntimeError(f'{self.__class__.__name__} must be trained before saving')

        if not filepath.endswith('.pkl'):
            raise ValueError('only .pkl format is supported')

        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)

        except Exception as e:
            print("Failed to save the model:", e)

    @staticmethod
    def load(filepath):
        """
        Loads a trained model from a file.

        :param filepath: path to the saved model file
        :return: the loaded Model instance
        """
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print("Failed to load the model:", e)