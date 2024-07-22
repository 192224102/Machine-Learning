import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
degree = 3
poly_features = PolynomialFeatures(degree=degree)
poly_model = make_pipeline(poly_features, LinearRegression())
poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)
print("Linear Regression:")
print("Mean Squared Error:", mse_linear)
print("R-squared:", r2_linear)
print("\nPolynomial Regression:")
print("Mean Squared Error:", mse_poly)
print("R-squared:", r2_poly)
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, linear_model.predict(X), color='red', linewidth=2, label='Linear fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(x_range, poly_model.predict(x_range), color='red', linewidth=2, label='Polynomial fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'Polynomial Regression (Degree={degree})')
plt.legend()
plt.show()
