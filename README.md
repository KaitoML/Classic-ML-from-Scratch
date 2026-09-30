# Classic ML from Scratch

An educational machine learning framework implemented from scratch in pure NumPy. The goal is to understand how classical ML algorithms work internally by building them without relying on high-level libraries such as scikit-learn.

The project is under active development.

## Features

- **Models** (`models.py`) — `LinearRegression` (gradient descent with L1/Lasso and L2/Ridge regularization, early stopping), `LogisticRegression` (binary classification), `KNNClassifier` (multiclass-capable).
- **Preprocessors** (`preprocessors.py`) — `StandardScaler`, `MinMaxScaler`, `PolynomialFeatures`.
- **Pipeline** (`pipeline.py`) — chain preprocessors and an optional final model in a single object.
- **Model selection** (`model_selection.py`) — `random_split` (train/test split) and `cv_score` (k-fold cross-validation).
- **Metrics** (`metrics.py`) — `r_squared`, `accuracy`, `precision`, `recall`, `f1`, `show_confusion_matrix`.
- **Utils** (`utils.py`) — `make_regression` and `make_classification` for generating synthetic datasets.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.x and NumPy. To run `tests.ipynb` you will also need Jupyter (or an IDE with notebook support). Matplotlib and seaborn are used for plotting the confusion matrix.

## Usage

```python
import numpy as np
from models import LinearRegression, LogisticRegression, KNNClassifier
from preprocessors import StandardScaler, PolynomialFeatures
from pipeline import Pipeline
from model_selection import random_split, cv_score
from metrics import accuracy, r_squared
from utils import make_regression, make_classification

# regression with polynomial features
x, y = make_regression(n_samples=500, n_features=3, noise_coef=0.1, seed=42)
x_train, x_test, y_train, y_test = random_split(x, y, test_size=0.2, seed=42)

pipe = Pipeline([
    PolynomialFeatures(degree=2),
    StandardScaler(),
    LinearRegression(max_iter=2000, lr=1e-2, ridge_coef=0.01)
])
losses = pipe.fit(x_train, y_train)
y_pred = pipe(x_test)
print("R²:", r_squared(y_test, y_pred))

# binary classification
x, y = make_classification(n_samples=1000, n_features=5, n_classes=2, seed=42)
clf = LogisticRegression(lr=0.1, max_iter=3000)
scores = cv_score(x, y, model=clf, metric=accuracy, n_folds=5, return_value='mean')
print("CV accuracy:", scores)

# multiclass with KNN
x, y = make_classification(n_samples=1000, n_features=10, n_classes=5, seed=42)
x_train, x_test, y_train, y_test = random_split(x, y, test_size=0.2, seed=42)

knn = KNNClassifier(n_neighbors=5)
knn.train(x_train, y_train)
print("Accuracy:", accuracy(y_test, knn(x_test)))
```

See `tests.ipynb` for more complete examples.

## Motivation

Built as a long-term learning project to deeply understand the mathematical foundations and optimization procedures behind classical machine learning algorithms.

## License

This project is licensed under the MIT License.