import numpy as np
from .base import Model

class Node:
    """
    Single node of a decision tree.

    :param feature_idx: index of the feature used for splitting
    :param threshold: threshold value for the split
    :param labels: labels that reached this node during training
    :param prediction: prediction value if the node is a leaf
    """

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
    """
    Base CART-style decision tree.
    Used as a parent for classifier and regressor variants.

    :param task: "classification" or "regression"
    :param max_depth: maximum depth of the tree
    :param max_features: number of features to consider at each split ("all" or "sqrt")
    :param random_thresholds: if True, thresholds are sampled randomly (ExtraTrees-style)
    :param seed: random seed for reproducibility
    """

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
        self.amount_of_say = None # for AdaBoost

    def _criterion(self, y):
        """
        Calculates impurity or variance for a set of labels.

        :param y: labels in the current node
        :return: criterion value
        """
        raise NotImplementedError(f'_criterion for {self.__class__.__name__} is not defined')

    def _prediction(self, y):
        """
        Calculates the prediction for a leaf node.

        :param y: labels in the leaf
        :return: leaf prediction
        """
        raise NotImplementedError(f'_prediction for {self.__class__.__name__} is not defined')

    def _build_tree(self, x, y, depth):
        """
        Recursively builds the decision tree.

        :param x: input features for the current node
        :param y: labels for the current node
        :param depth: current depth of the node
        :return: root Node of the (sub)tree
        """

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
        """
        Builds the decision tree on the given data.

        :param x: input features of shape (n_samples, n_features)
        :param y: target labels of shape (n_samples,)
        :return: None
        """
        if self.task == 'classification':
            y = np.asarray(y).astype(np.int64)

        self._rng = np.random.default_rng(seed=self.seed)

        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)

        self.root = self._build_tree(x, y, depth=0)
        self.trained = True

    def _traverse(self, sample, node):
        """
        Traverses the tree to obtain a prediction for a single sample.

        :param sample: single input sample
        :param node: current node in the tree
        :return: prediction from the reached leaf
        """
        if node.leaf:
            return node.prediction

        if sample[node.feature_idx] < node.threshold:
            return self._traverse(sample, node.left)
        else:
            return self._traverse(sample, node.right)

    def __call__(self, x):
        """
        Makes predictions by traversing the trained tree.

        :param x: input features of shape (n_samples, n_features) or (n_features,)
        :return: predicted values
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)

        preds = []
        for x_i in x:
            pred = self._traverse(x_i, self.root)
            preds.append(pred)

        return np.array(preds)

class DecisionTreeClassifier(DecisionTree):
    """
    Decision tree classifier using Gini impurity (CART).

    :param max_depth: maximum depth of the tree
    :param max_features: number of features to consider at each split ("all" or "sqrt")
    :param random_thresholds: if True, thresholds are sampled randomly (ExtraTrees-style)
    :param seed: random seed for reproducibility
    """

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
        """
        Calculates Gini impurity.

        :param y: labels in the current node
        :return: Gini impurity
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _prediction(self, y):
        """
        Returns the majority class in the leaf.

        :param y: labels in the leaf
        :return: majority class label
        """
        classes, counts = np.unique(y, return_counts=True)
        return classes[counts.argmax()]

class DecisionTreeRegressor(DecisionTree):
    """
    Decision tree regressor using mean squared error (CART).

    :param max_depth: maximum depth of the tree
    :param max_features: number of features to consider at each split ("all" or "sqrt")
    :param random_thresholds: if True, thresholds are sampled randomly (ExtraTrees-style)
    :param seed: random seed for reproducibility
    """

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
        """
        Calculates mean squared error (variance) of the labels.

        :param y: labels in the current node
        :return: mean squared error
        """
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def _prediction(self, y):
        """
        Returns the mean of the labels in the leaf.

        :param y: labels in the leaf
        :return: mean value
        """
        return np.mean(y)

class XGBoostRegressionTree(Model):
    def __init__(self, max_depth=5, lbd=0, gamma=0, seed=None):
        super().__init__(task='regression')
        self.max_depth = max_depth
        self.lbd = lbd
        self.gamma = gamma
        self.seed = seed

        self.root = None

    def __repr__(self):
        return (f'XGBoostRegressionTree('
                f'\n    max_depth={self.max_depth},'
                f'\n    lbd={self.lbd}'
                f'\n    gamma={self.gamma}'
                f'\n    trained={self.trained}'
                f'\n    seed={self.seed}'
                f'\n)')

    def _similarity_score(self, resids):
        return np.square(np.sum(resids)) / (len(resids) + self.lbd)

    def _prediction(self, resids):
        return np.sum(resids) / (len(resids) + self.lbd)

    def _build_tree(self, x, resids, depth):

        # if max depth is reached
        if depth >= self.max_depth:
            node = Node(
                labels=resids,
                prediction=self._prediction(resids)
            )
            node.leaf = True
            return node

        candidate_splits = {}  # {feature_idx: (gain, threshold)}

        feature_indices = np.arange(x.shape[1])

        for i in feature_indices:
            feature = x[:, i]
            unique_vals = np.unique(feature)

            if len(unique_vals) < 2:
                continue

            thresholds = []
            sorted_values = sorted(feature)
            for val1, val2 in zip(sorted_values, sorted_values[1:]):
                if val1 != val2:
                    local_avg = (val1 + val2) / 2
                    thresholds.append(local_avg)

            gains = {}
            for t in thresholds:
                left = []
                right = []

                for sample, resid in zip(x, resids):
                    if sample[i] < t:
                        left.append(resid)
                    else:
                        right.append(resid)

                if len(left) == 0 or len(right) == 0:
                    continue

                similarity_root = self._similarity_score(resids)
                similarity_left = self._similarity_score(left)
                similarity_right = self._similarity_score(right)
                gain = similarity_left + similarity_right - similarity_root - self.gamma

                gains[t] = gain

            if not gains:
                continue

            best_threshold = max(gains, key=gains.get)
            highest_gain = gains[best_threshold]

            if highest_gain <= 0:
                node = Node(
                    labels=resids,
                    prediction=self._prediction(resids)
                )
                node.leaf = True
                return node

            candidate_splits[i] = (highest_gain, best_threshold)

        if not candidate_splits:
            node = Node(
                labels=resids,
                prediction=self._prediction(resids)
            )
            node.leaf = True
            return node

        best_feature, (highest_gain, best_threshold) = max(candidate_splits.items(), key=lambda item: item[1][0])
        node = Node(feature_idx=best_feature, threshold=best_threshold, labels=resids)

        left_x = []
        left_resid = []
        right_x = []
        right_resid = []
        for sample, resid in zip(x, resids):
            if sample[best_feature] < best_threshold:
                left_x.append(sample)
                left_resid.append(resid)
            else:
                right_x.append(sample)
                right_resid.append(resid)

        left_x = np.array(left_x)
        left_resid = np.array(left_resid)
        right_x = np.array(right_x)
        right_resid = np.array(right_resid)

        node.left = self._build_tree(left_x, left_resid, depth + 1)
        node.right = self._build_tree(right_x, right_resid, depth + 1)

        return node

    def train(self, x, resids):
        if x.ndim == 1:
            x = np.expand_dims(x, axis=1)

        self.root = self._build_tree(x, resids, depth=0)
        self.trained = True

    def _traverse(self, x_i, node):
        if node.leaf:
            return node.prediction

        if x_i[node.feature_idx] < node.threshold:
            return self._traverse(x_i, node.left)
        else:
            return self._traverse(x_i, node.right)

    def __call__(self, x):
        if not self.trained:
            raise RuntimeError(f'{self.__class__.__name__} is not trained')

        preds = []
        for x_i in x:
            pred = self._traverse(x_i, self.root)
            preds.append(pred)

        return np.array(preds)