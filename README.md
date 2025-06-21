# Polynomial Regression with Regularization

This project demonstrates polynomial regression using synthetic data, along with the application of L2 (Ridge) and L1 (Lasso) regularization techniques. It highlights model fitting, performance evaluation, and visualization to show the impact of regularization on overfitting and model complexity.

## Project Overview

- Generates synthetic nonlinear data (quadratic relationship with noise)
- Applies polynomial feature transformation
- Standardizes features to improve training stability
- Compares three models:
  - Linear Regression (no regularization)
  - Ridge Regression (L2)
  - Lasso Regression (L1)
- Visualizes model predictions and compares their mean squared error (MSE)
- Outputs model coefficients and intercepts for interpretability

## Technologies Used

- Python 3
- NumPy
- Matplotlib
- Scikit-learn (Pipeline, PolynomialFeatures, LinearRegression, Ridge, Lasso)

## Key Concepts Demonstrated

- Polynomial regression
- Regularization techniques (Ridge and Lasso)
- Feature scaling and preprocessing pipelines
- Model evaluation using Mean Squared Error (MSE)
- Visual comparison of model fits

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Polynomial-Regression-with-Regularization.git
