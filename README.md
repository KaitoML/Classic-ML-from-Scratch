# Classic ML from Scratch

An educational machine learning framework implemented from scratch in pure NumPy. The goal is to understand how classical ML algorithms work internally by building them without relying on high-level libraries such as scikit-learn.

The project is under active development.

## Features

- **Models** (`models.py`) — `LinearRegression` (gradient descent with L1/Lasso and L2/Ridge regularization, early stopping), `LogisticRegression` (binary classification), `KNNClassifier`.
- **Preprocessors** (`preprocessors.py`) — `StandardScaler`, `MinMaxScaler`, and a basic `Pipeline` sketch for chaining preprocessors and models.
- **Model selection** (`model_selection.py`) — `random_split` (train/test split) and `cv_score` (k-fold cross-validation).
- **Metrics** (`metrics.py`) — `accuracy`, `precision`, `recall`, `f1`.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.x and NumPy. To run `tests.ipynb` you will also need Jupyter (or an IDE with notebook support).

## Usage

```python
import numpy as np
from models import LinearRegression, LogisticRegression
from preprocessors import StandardScaler
from model_selection import random_split, cv_score
from metrics import accuracy

# regression example
X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
y = np.array([3.0, 5.0, 7.0, 9.0])

model = LinearRegression(max_iter=1000, lr=1e-2, ridge_coef=0.01)
losses = model.train(X, y)
predictions = model(X)

# classification + cross-validation example
X_train, X_test, y_train, y_test = random_split(X, y, test_size=0.25, seed=42)

clf = LogisticRegression(lr=0.1)
scores = cv_score(X, y, model=clf, metric=accuracy, n_folds=3, return_value='all')
print(scores)
```

See `tests.ipynb` for more complete examples.

## Motivation

Built as a long-term learning project to deeply understand the mathematical foundations and optimization procedures behind classical machine learning algorithms.

## License

This project is licensed under the MIT License.
```