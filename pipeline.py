from models import Model
from preprocessors import Preprocessor

class Pipeline:
    def __init__(self, components: list | tuple):
        if not isinstance(components, (list, tuple)):
            raise TypeError('components must be list or tuple')

        self.components = list(components)

        if len(self.components) < 1:
            raise ValueError('the pipeline is empty')

        model_indices = [i for i in range(len(self.components)) if isinstance(self.components[i], Model)]
        if model_indices and len(model_indices) > 1:
            raise ValueError('pipeline supports one model at most')

        if model_indices and not isinstance(self.components[-1], Model):
            raise ValueError('model can only be at the end of the pipeline')

    def __repr__(self):
        return f'Pipeline({self.components})'

    def fit(self, x, y=None):
        """
        Fit all preprocessors and (optionally) train the final model.
        Returns losses if the final component is a model, otherwise returns None

        :param x: input data
        :param y: labels
        :return: losses | None
        """
        losses = None

        for c in self.components:
            if isinstance(c, Preprocessor):
                x = c.fit_transform(x)

            elif isinstance(c, Model):
                if y is None:
                    raise ValueError('you must pass y to train a model')

                losses = c.train(x, y)

            else:
                raise TypeError(f'{c} <-- is not compatible with Pipeline')

        return losses

    def __call__(self, x):
        """
        Transform data through all preprocessors and (optionally) get predictions if the pipeline includes a model

        :param x: input data
        :return: predictions | transformed data
        """

        for c in self.components:
            if isinstance(c, Preprocessor):
                x = c.transform(x)

            elif isinstance(c, Model):
                return c(x)

            else:
                raise TypeError(f'{c} <-- is not compatible with Pipeline')

        return x