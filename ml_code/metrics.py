import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# REGRESSION METRICS:
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculated mean squared error

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: mse score
    """
    score = np.mean((y_true - y_pred) ** 2)
    return score

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculated root mean squared error

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: mse score
    """
    score = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return score

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

def residuals_plot(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Displays residuals plot

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: None
    """
    resids = y_true - y_pred
    plt.scatter(y_true, resids, alpha=0.75)
    plt.axhline(0, ls='--', c='k')
    plt.title('Residuals plot')
    plt.xlabel('True values')
    plt.ylabel('Residuals')
    plt.show()

# CLASSIFICATION METRICS:
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

    if (p + r) == 0:
        return 0.0

    return 2 * p * r / (p + r)

def show_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Displays confusion matrix

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: None
    """
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