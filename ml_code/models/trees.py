import numpy as np
from .base import Model

class Node:
    def __init__(self, feature_idx=None, threshold=None, labels=None, prediction=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.labels = labels
        self.prediction = prediction

        self.right = None
        self.left = None
        self.leaf = False

    def __repr__(self):
        return f'Node({self.feature_idx, self.threshold, self.labels})'

class DecisionTree(Model):
    def __init__(self, task, max_depth=5, max_features='all', random_thresholds=False, seed=None):
        super().__init__(task=task)
        if max_features not in ('all', 'sqrt'):
            raise ValueError('max_features can only be "all" or "sqrt". ')

        self.task = task
        self.random_thresholds = random_thresholds # for extra-trees
        self.max_features = max_features
        self.max_depth = max_depth
        self.root = None
        self.seed = seed
        self._rng = None

    def _criterion(self, y):
        raise NotImplementedError(f'_criterion for {self.__class__.__name__} is not defined')

    def _prediction(self, y):
        raise NotImplementedError(f'_prediction for {self.__class__.__name__} is not defined')

    def _build_tree(self, x, y, depth):

        # if max depth is reached
        if depth >= self.max_depth:
            prediction = self._prediction(y)
            node = Node(labels=y, prediction=prediction)
            node.leaf = True
            return node

        # if node is pure
        if len(np.unique(y)) == 1:
            node = Node(
                labels=y,
                prediction=self._prediction(y)
            )
            node.leaf = True
            return node

        candidate_splits = {}  # {feature_idx: (criterion, threshold)}

        if self.max_features == 'sqrt':
            n_features = x.shape[1]
            target_n_features = int(np.sqrt(n_features))
            feature_indices = self._rng.choice(x.shape[1], size=target_n_features, replace=False)
        else:
            feature_indices = np.arange(x.shape[1])

        for i in feature_indices:
            feature = x[:, i]
            unique_vals = np.unique(feature)

            if len(unique_vals) < 2:
                continue

            if self.random_thresholds:
                t = self._rng.uniform(unique_vals.min(), unique_vals.max())
                thresholds = [t]
            else:
                thresholds = []

                sorted_values = sorted(feature)
                for val1, val2 in zip(sorted_values, sorted_values[1:]):
                    if val1 != val2:
                        local_avg = (val1 + val2) / 2
                        thresholds.append(local_avg)

            criterion_vals = {}
            for t in thresholds:
                left = []
                right = []

                for sample, label in zip(x, y):
                    if sample[i] < t:
                        left.append(label)
                    else:
                        right.append(label)

                # variance / impurity for each child
                criterion_left = self._criterion(left)
                criterion_right = self._criterion(right)

                total_samples = len(left) + len(right)

                # variance / impurity for split
                weighted_criterion = (
                        len(left) / total_samples * criterion_left
                        + len(right) / total_samples * criterion_right
                )

                criterion_vals[t] = weighted_criterion

            if not criterion_vals:
                continue

            best_threshold = min(criterion_vals, key=criterion_vals.get)
            lowest_criterion = criterion_vals[best_threshold]
            candidate_splits[i] = (lowest_criterion, best_threshold)

        # if impossible to split further
        if not candidate_splits:
            prediction = self._prediction(y)
            node = Node(labels=y, prediction=prediction)
            node.leaf = True
            return node

        best_feature, (lowest_crit, best_threshold) = min(candidate_splits.items(), key=lambda item: item[1][0])
        node = Node(feature_idx=best_feature, threshold=best_threshold, labels=y)

        left_x = []
        left_y = []
        right_x = []
        right_y = []
        for sample, label in zip(x, y):
            if sample[best_feature] < best_threshold:
                left_x.append(sample)
                left_y.append(label)
            else:
                right_x.append(sample)
                right_y.append(label)

        left_x = np.array(left_x)
        left_y = np.array(left_y)
        right_x = np.array(right_x)
        right_y = np.array(right_y)

        node.left = self._build_tree(left_x, left_y, depth + 1)
        node.right = self._build_tree(right_x, right_y, depth + 1)

        return node

    def train(self, x, y):
        if self.task == 'classification':
            y = np.asarray(y).astype(np.int64)

        self._rng = np.random.default_rng(seed=self.seed)

        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)

        self.root = self._build_tree(x, y, depth=0)
        self.trained = True

    def _traverse(self, sample, node):
        if node.leaf:
            return node.prediction

        if sample[node.feature_idx] < node.threshold:
            return self._traverse(sample, node.left)
        else:
            return self._traverse(sample, node.right)

    def __call__(self, x):
        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)

        preds = []
        for x_i in x:
            pred = self._traverse(x_i, self.root)
            preds.append(pred)

        return np.array(preds)

class DecisionTreeClassifier(DecisionTree):
    def __init__(self, max_depth=5, max_features='all', random_thresholds=False, seed=None):
        super().__init__(task='classification',
                         max_depth=max_depth,
                         max_features=max_features,
                         random_thresholds=random_thresholds,
                         seed=seed)

    def __repr__(self):
        return (f'DecisionTreeClassifier('
                f'\n    max_depth={self.max_depth},'
                f'\n    max_features={self.max_features},'
                f'\n    random_thresholds={self.random_thresholds},'
                f'\n    trained={self.trained}'
                f'\n    seed={self.seed}'
                f'\n)')

    def _criterion(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _prediction(self, y):
        classes, counts = np.unique(y, return_counts=True)
        return classes[counts.argmax()]

class DecisionTreeRegressor(DecisionTree):
    def __init__(self, max_depth=5, max_features='all', random_thresholds=False, seed=None):
        super().__init__(task='regression',
                         max_depth=max_depth,
                         max_features=max_features,
                         random_thresholds=random_thresholds,
                         seed=seed)

    def __repr__(self):
        return (f'DecisionTreeRegressor('
                f'\n    max_depth={self.max_depth},'
                f'\n    max_features={self.max_features},'
                f'\n    random_thresholds={self.random_thresholds},'
                f'\n    trained={self.trained}'
                f'\n    seed={self.seed}'
                f'\n)')

    def _criterion(self, y):
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def _prediction(self, y):
        return np.mean(y)