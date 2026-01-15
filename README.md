# Simple Linear Regression from Scratch

A minimal Python class implementing simple linear regression using gradient descent.

## SimpleR Class

- `__init__(lr=0.1, max_iter=200, threshold=1e-6)`: Initialize model with learning rate, max iterations, and convergence threshold.  
- `predict(X)`: Predict output for input array `X`.  
- `fit(X, Y)`: Train model on data `(X, Y)` using gradient descent; returns loss history.  
- `plot(X, Y)`: Plot data points and fitted regression line.

## Formulae Used

- Hypothesis:
  ŷ = w·x + b

- Loss (Mean Squared Error):
  J(w,b) = (1 / (2n)) · Σ (y_i - ŷ_i)^2

- Gradients:
  ∂J/∂w = -(1 / n) · Σ (y_i - ŷ_i) x_i  
  ∂J/∂b = -(1 / n) · Σ (y_i - ŷ_i)

- Parameter update (as implemented):
  w := w - α ∂J/∂w = w + α · (1 / n) · Σ (y_i - ŷ_i) x_i  
  b := b - α ∂J/∂b = b + α · (1 / n) · Σ (y_i - ŷ_i)

(Here α is the learning rate.)

## Usage Example

```python
import numpy as np
from simple_linear_regression import SimpleR

X = np.array([1.1, 2.0, 3.2, 4.5, 5.1])
Y = np.array([39000, 48000, 60000, 80000, 90000])

model = SimpleR(lr=0.01, max_iter=1000)
loss = model.fit(X, Y)
model.plot(X, Y)
