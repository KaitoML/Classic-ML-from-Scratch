import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculated R-squared

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: r-squared score
    """
    rss = np.sum(np.square(y_true - y_pred)) # unexplained variance
    tss = np.sum(np.square(y_true - np.mean(y_true))) # total variance
    score = 1 - rss / tss
    return score

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

def show_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n_classes = max(max(y_true), max(y_pred)) + 1
    confmat = np.zeros((n_classes, n_classes))

    for true, pred in zip(y_true, y_pred):
        confmat[true, pred] += 1

    sns.heatmap(confmat,
                annot=True,
                cmap="rocket")

    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

if __name__ == '__main__':
    y_tr = np.array([1, 0, 1])
    y_pr = np.array([1, 1, 1])

    print(show_confusion_matrix(y_tr, y_pr))