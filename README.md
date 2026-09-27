# Classic ML from Scratch

An educational machine learning framework implemented from scratch using Python and NumPy.

The main goal of this project is to understand how classical machine learning algorithms work internally by implementing them without relying on high-level ML libraries such as scikit-learn.

The project is currently under active development.

## Implemented

* Linear Regression

  * Gradient descent
  * L1 regularization (Lasso)
  * L2 regularization (Ridge)
* Logistic Regression

  * Binary classification
  * Gradient descent
* K-Nearest Neighbors

## Planned

The framework will (hopefully) include implementations of different classical machine learning approaches, including:

* K-Nearest Neighbors
* Decision Trees
* Random Forest
* Naive Bayes
* Support Vector Machines
* Clustering algorithms
* Ensemble methods
* Additional regression and classification algorithms

The exact scope may change as the project develops.

## Philosophy

This is primarily a learning project. The implementations are intentionally kept relatively simple and transparent rather than optimized for production use.

## Example

```python
import numpy as np
from models import LinearRegression

X = np.array([
    [1, 1.2, 1.3],
    [2.4, 3.5, 2.2]
])

y = np.array([16.7, 34])

model = LinearRegression(
    max_iter=1000,
    lr=1e-3,
    ridge_coef=0.0001
)

losses = model.train(X, y)

print(model(X))
print(model)
```
