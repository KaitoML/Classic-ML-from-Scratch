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
│       ├── bayes.py
│       ├── clustering.py
│       ├── boosting.py
│       ├── ensembles.py
│       └── __init__.py
├── showcases/
├── requirements.txt
├── README.md
└── LICENSE
```

## Features

- **Models** (`ml_code/models/`)
  - `LinearRegression` — gradient descent with L1/Lasso and L2/Ridge regularization, early stopping
  - `LogisticRegression` — binary classification (sigmoid + binary cross-entropy)
  - `KNNClassifier` — multiclass-capable k-nearest neighbors (Euclidean distance)
  - `GaussianNaiveBayes` — Gaussian Naive Bayes classifier
  - `DecisionTreeClassifier` / `DecisionTreeRegressor` — CART-style trees (Gini / MSE), support for `max_features` and random thresholds (ExtraTrees-style)
  - `AdaBoostClassifier` — AdaBoost with decision stumps and weighted voting
  - `GradientBoostRegressor` — gradient boosting for squared error (trees on residuals + learning rate)
  - `GradientBoostClassifier` — binary gradient boosting with logistic loss (log-odds space + sigmoid)
  - `VotingClassifier` / `VotingRegressor` — hard voting / averaging over multiple models
  - `BaggingClassifier` / `BaggingRegressor` — bootstrap aggregating
  - `RandomForestClassifier` / `RandomForestRegressor` — bagging of decision trees with feature subsampling
  - `ExtraTreesClassifier` / `ExtraTreesRegressor` — extremely randomized trees
  - `KMeans` — K-Means clustering with multiple initializations and inertia-based selection
- **Preprocessors** (`ml_code/preprocessors.py`) — `StandardScaler`, `MinMaxScaler`, `PolynomialFeatures`
- **Pipeline** (`ml_code/pipeline.py`) — chain preprocessors and an optional final model
- **Model selection** (`ml_code/model_selection.py`) — `random_split`, `cv_score` (k-fold cross-validation), `GridSearchCV`
- **Metrics** (`ml_code/metrics.py`) — `mse`, `rmse`, `r_squared`, `accuracy`, `precision`, `recall`, `f1`, `show_confusion_matrix`
- **Utils** (`ml_code/utils.py`) — `make_regression`, `make_classification`

## Installation

```bash
pip install -r requirements.txt
```

Requires **Python 3.x** and **NumPy**.  
`matplotlib` and `seaborn` are needed for the confusion-matrix plot.  
To run the example notebooks you will also need Jupyter (or any IDE with notebook support).

## Usage

```python
from ml_code.models import LinearRegression
from ml_code.preprocessors import StandardScaler
from ml_code.pipeline import Pipeline
from ml_code.model_selection import random_split
from ml_code.metrics import r_squared
from ml_code.utils import make_regression

x, y = make_regression(n_samples=500, n_features=5, noise_coef=0.1, seed=42)
x_train, x_test, y_train, y_test = random_split(x, y, test_size=0.2, seed=42)

pipe = Pipeline([
    StandardScaler(),
    LinearRegression(max_iter=2000, lr=1e-2, ridge_coef=0.01),
])
pipe.fit(x_train, y_train)
y_pred = pipe(x_test)

print("R²:", r_squared(y_test, y_pred))
```

See `showcases/` for more complete examples covering other models.

## Motivation

Built as a long-term learning project to deeply understand the mathematical foundations and optimization procedures behind classical machine learning algorithms.

## Notes

Core algorithm implementations (models, training logic, ensembles, trees, preprocessors, pipeline, model selection, metrics, and data utilities) were written independently.  
Documentation (docstrings) and parts of the README were prepared with AI assistance.

## License

This project is licensed under the MIT License.
