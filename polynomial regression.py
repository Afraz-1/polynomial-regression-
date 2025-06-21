import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

# Step 1: Generate synthetic data
np.random.seed(42)
X = 6 * np.random.rand(100, 1) - 3  # X values between -3 and 3
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)  # Quadratic + noise

# Step 2: Visualize raw data
plt.scatter(X, y, color='blue', alpha=0.5)
plt.title("Raw Data")
plt.xlabel("X")
plt.ylabel("y")
plt.grid(True)
plt.show()

# Step 3: Set up models using pipelines (Polynomial + Scaler + Regression)
degree = 2

models = {
    "Linear": Pipeline([
        ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("lin_reg", LinearRegression())
    ]),
    "Ridge": Pipeline([
        ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge_reg", Ridge(alpha=1.0))
    ]),
    "Lasso": Pipeline([
        ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("lasso_reg", Lasso(alpha=0.1))
    ])
}

# Step 4: Train models and evaluate
x_new = np.linspace(-3, 3, 100).reshape(100, 1)
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='gray', alpha=0.5, label='Data')

for name, model in models.items():
    model.fit(X, y.ravel())
    y_pred = model.predict(x_new)
    mse = mean_squared_error(y, model.predict(X))
    coeffs = model.named_steps[list(model.named_steps.keys())[-1]].coef_
    intercept = model.named_steps[list(model.named_steps.keys())[-1]].intercept_

    print(f"{name} Regression")
    print("  Coefficients:", coeffs)
    print("  Intercept:", intercept)
    print("  MSE:", round(mse, 4), "\n")

    plt.plot(x_new, y_pred, label=f"{name} (MSE={round(mse, 2)})")

plt.title("Polynomial Regression with Regularization")
plt.xlabel("X")
plt.ylabel("Predicted y")
plt.legend()
plt.grid(True)
plt.show()
