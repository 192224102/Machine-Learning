import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# Creating a synthetic mobile price dataset using pandas DataFrame
data = {
    'battery_power': [1000, 1500, 2000, 2500, 3000],
    'ram': [2, 3, 4, 4, 6],
    'internal_memory': [16, 32, 64, 128, 256],
    '3g': [0, 1, 1, 1, 1],
    '4g': [1, 0, 1, 0, 1],
    'price_range': ['low', 'medium', 'medium', 'high', 'high']
}
df = pd.DataFrame(data)
# Splitting the data into features (X) and target (y)
X = df.drop('price_range', axis=1)
y = df['price_range']
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the Decision Tree classifier
model = DecisionTreeClassifier()
# Training the model
model.fit(X_train, y_train)
# Making predictions
y_pred = model.predict(X_test)

# Calculating accuracy (for demonstration purposes)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Decision Tree classifier: {accuracy}")
# Example prediction for a new data point
new_mobile = pd.DataFrame({
    'battery_power': [1800],
    'ram': [3],
    'internal_memory': [32],
    '3g': [1],
    '4g': [0]
})
predicted_price_range = model.predict(new_mobile)
print(f"Predicted price range for new mobile: {predicted_price_range}")
