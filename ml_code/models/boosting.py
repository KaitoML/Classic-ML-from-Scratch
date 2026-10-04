import numpy as np
from .base import Model
from .trees import DecisionTreeClassifier

class AdaBoostClassifier(Model):
    def __init__(self, n_models=50, seed=None):
        super().__init__(task='classification')
        self.n_models = n_models
        self.seed = seed

        self._rng = np.random.default_rng(self.seed)
        self.stumps = []

    def __repr__(self):
        return (f'AdaBoost('
                f'\n   n_models={self.n_models},'
                f'\n   seed={self.seed},'
                f'\n   trained={self.trained}'
                f'\n)')

    def train(self, x, y):
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 1)

        n_samples = x.shape[0]
        sample_weights = np.full(x.shape[0], fill_value=1/n_samples)

        for _ in range(self.n_models):

            indices = self._rng.choice(
                n_samples,
                size=n_samples,
                p=sample_weights
            )

            data = x[indices]
            labels = y[indices]

            stump = DecisionTreeClassifier(max_depth=1, seed=self.seed)
            stump.train(data, labels)
            pred = stump(x)
            self.stumps.append(stump)

            # estimate error for a stump
            eps = 1e-12
            total_error = np.sum(sample_weights[pred != y])
            total_error = np.clip(total_error, eps, 1 - eps)

            # estimate amount of say for a stump
            stump.amount_of_say = 0.5 * np.log((1 - total_error)/total_error)

            # update sample weights
            # if error: *= exp(amount of say)
            # if correct: *= exp(-amount of say)
            sample_weights *= np.exp(stump.amount_of_say * (pred != y).astype(float) * 2 - stump.amount_of_say)

            # scale so that they add up to one
            sample_weights /= np.sum(sample_weights)

        self.trained = True

    def __call__(self, x):
        stumps_preds = []

        for s in self.stumps:
            pred = s(x)
            stumps_preds.append(pred)

        stumps_preds = np.array(stumps_preds)

        outs = []

        for options in stumps_preds.T:
            weighted_voting = {} # {class: total say for this class}
            for i, option in enumerate(options):
                weighted_voting[option] = weighted_voting.get(option, 0) + self.stumps[i].amount_of_say

            outs.append(max(weighted_voting, key=weighted_voting.get))

        return np.array(outs)