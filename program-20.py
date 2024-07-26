import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Example dataset
data = {
    'month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'sales': [100, 120, 130, 140, 160, 180, 200, 210, 220, 230]
}
df = pd.DataFrame(data)

# Splitting the data into features (X) and target (y)
X = df[['month']]
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)
 
y_pred = model.predict(X_test)

# Calculating mean squared error (for demonstration purposes)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Example prediction for future month (assuming month 11)
future_month = pd.DataFrame({'month': [11]})
predicted_sales = model.predict(future_month)
print(f"Predicted sales for future month: {predicted_sales[0]}")
