# Classic ML from Scratch

An educational machine learning framework implemented from scratch in pure NumPy. The goal is to understand how classical ML algorithms work internally by building them without relying on high-level libraries such as scikit-learn.

The project is under active development.

## Project structure

```text
├── ml_code/
│   ├── metrics.py
│   ├── model_selection.py
│   ├── pipeline.py
│   ├── preprocessors.py
│   ├── utils.py
│   └── models/
│       ├── base.py
│       ├── linear.py
│       ├── neighbors.py
│       ├── trees.py
│       ├── ensembles.py
│       └── __init__.py
├── tests.ipynb
├── requirements.txt
├── README.md
└── LICENSE
```

## Features

- **Models** (`ml_code/models/`)
  - `LinearRegression` — gradient descent with L1/Lasso and L2/Ridge regularization, early stopping
  - `LogisticRegression` — binary classification (sigmoid + binary cross-entropy)
  - `KNNClassifier` — multiclass-capable k-nearest neighbors (Euclidean distance)
  - `DecisionTreeClassifier` / `DecisionTreeRegressor` — CART-style trees (Gini / MSE), support for `max_features` and random thresholds (ExtraTrees-style)
  - `VotingClassifier` / `VotingRegressor` — hard voting / averaging over multiple models
  - `BaggingClassifier` / `BaggingRegressor` — bootstrap aggregating
  - `RandomForestClassifier` / `RandomForestRegressor` — bagging of decision trees with feature subsampling
  - `ExtraTreesClassifier` / `ExtraTreesRegressor` — extremely randomized trees
- **Preprocessors** (`ml_code/preprocessors.py`) — `StandardScaler`, `MinMaxScaler`, `PolynomialFeatures`
- **Pipeline** (`ml_code/pipeline.py`) — chain preprocessors and an optional final model
- **Model selection** (`ml_code/model_selection.py`) — `random_split`, `cv_score` (k-fold cross-validation)
- **Metrics** (`ml_code/metrics.py`) — `r_squared`, `accuracy`, `precision`, `recall`, `f1`, `show_confusion_matrix`
- **Utils** (`ml_code/utils.py`) — `make_regression`, `make_classification`

## Installation

```bash
pip install -r requirements.txt
```

Requires **Python 3.x** and **NumPy**.  
`matplotlib` and `seaborn` are needed for the confusion-matrix plot.  
To run `tests.ipynb` you will also need Jupyter (or any IDE with notebook support).

## Usage

```python
from ml_code.models import (
    LinearRegression, LogisticRegression, KNNClassifier,
    DecisionTreeClassifier, DecisionTreeRegressor,
    BaggingClassifier, VotingClassifier,
    RandomForestClassifier, ExtraTreesClassifier,
)
from ml_code.preprocessors import StandardScaler, PolynomialFeatures
from ml_code.pipeline import Pipeline
from ml_code.model_selection import random_split, cv_score
from ml_code.metrics import accuracy, r_squared
from ml_code.utils import make_regression, make_classification

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

# bagging classifier
x, y = make_classification(n_samples=1000, n_features=5, n_classes=2, seed=42)
x_train, x_test, y_train, y_test = random_split(x, y, test_size=0.2, seed=42)

clf = BaggingClassifier(model=KNNClassifier(n_neighbors=5), n_models=15, samples=0.75, seed=42)
clf.train(x_train, y_train)
print("Accuracy:", accuracy(y_test, clf(x_test)))

# voting over different models
voter = VotingClassifier([LogisticRegression(lr=0.1), KNNClassifier(n_neighbors=3)])
voter.train(x_train, y_train)
print("Voting accuracy:", accuracy(y_test, voter(x_test)))

# random forest / extra trees
rf = RandomForestClassifier(n_models=30, max_depth=6, seed=42)
rf.train(x_train, y_train)
print("RF accuracy:", accuracy(y_test, rf(x_test)))

et = ExtraTreesClassifier(n_models=30, max_depth=6, seed=42)
et.train(x_train, y_train)
print("ExtraTrees accuracy:", accuracy(y_test, et(x_test)))
```

See `tests.ipynb` for more complete examples.

## Motivation

Built as a long-term learning project to deeply understand the mathematical foundations and optimization procedures behind classical machine learning algorithms.

## License

This project is licensed under the MIT License.
