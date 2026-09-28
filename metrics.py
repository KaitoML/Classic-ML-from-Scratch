import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates accuracy score

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: accuracy score
    """

    correct = (y_true == y_pred)
    total = len(y_true)
    score = np.sum(correct) / total
    return score

def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates precision

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: precision score
    """
    classes = np.unique(y_true)
    precisions = []

    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))

        if (tp + fp) == 0:
            continue

        score = tp / (tp + fp)
        precisions.append(score)

    return np.mean(precisions) if precisions else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates recall

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: recall score
    """
    classes = np.unique(y_true)
    recalls = []

    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        if (tp + fn) == 0:
            continue

        score = tp / (tp + fn)
        recalls.append(score)

    return np.mean(recalls) if recalls else 0.0

def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates f1-score

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: f1 score
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r)