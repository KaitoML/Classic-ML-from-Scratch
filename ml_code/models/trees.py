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

class DecisionTreeClassifier(Model):
    def __init__(self, max_depth=5):
        super().__init__(task='classification')
        self.max_depth = max_depth
        self.root = None

    def __repr__(self):
        return (f'DecisionTreeClassifier('
                f'\n    max_depth={self.max_depth},'
                f'\n    trained={self.trained}'
                f'\n)')

    def _build_tree(self, x, y, depth):

        # if max depth is reached
        if depth >= self.max_depth:
            prediction = np.bincount(y).argmax()
            node = Node(labels=y, prediction=prediction)
            node.leaf = True
            return node

        # if all samples belong to the same class
        if len(np.unique(y)) == 1:
            node = Node(
                labels=y,
                prediction=y[0]
            )
            node.leaf = True
            return node


        candidate_splits = {}  # {feature_idx: (impurity, threshold)}

        for i, feature in enumerate(x.T):

            if len(np.unique(feature)) == 2:
                thresholds = [np.mean(feature)]

            else:
                thresholds = []

                sorted_values = sorted(feature)
                for val1, val2 in zip(sorted_values, sorted_values[1:]):
                    if val1 != val2:
                        local_avg = (val1 + val2) / 2
                        thresholds.append(local_avg)

            impurity_vals = {}

            for t in thresholds:
                left = []
                right = []

                for sample, label in zip(x, y):
                    if sample[i] < t:
                        left.append(label)
                    else:
                        right.append(label)

                labels_count_left = {label: left.count(label) for label in left}
                labels_count_right = {label: right.count(label) for label in right}

                probs = []
                for v in labels_count_left.values():
                    p = (v / len(left)) ** 2
                    probs.append(p)
                impurity_left = 1
                for p in probs:
                    impurity_left -= p

                probs = []
                for v in labels_count_right.values():
                    p = (v / len(right)) ** 2
                    probs.append(p)
                impurity_right = 1
                for p in probs:
                    impurity_right -= p

                total_samples_split = len(right) + len(left)
                weighted_impurity_split = len(right) / total_samples_split * impurity_right + len(
                    left) / total_samples_split * impurity_left
                impurity_vals[t] = weighted_impurity_split

            t_lowest_imp = min(impurity_vals, key=impurity_vals.get)
            lowest_imp = impurity_vals[t_lowest_imp]
            candidate_splits[i] = (lowest_imp, t_lowest_imp)

        best_feature, (lowest_imp, best_threshold) = min(
            candidate_splits.items(), key=lambda item: item[1][0]
        )

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

        node.left = self._build_tree(left_x, left_y, depth+1)
        node.right = self._build_tree(right_x, right_y, depth+1)

        return node

    def train(self, x, y):
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
