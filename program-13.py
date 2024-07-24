import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
data = pd.DataFrame({
    'make': ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'Nissan', 'Toyota', 'Honda', 'Ford', 'Chevrolet', 'Nissan'],
    'model': ['Corolla', 'Civic', 'Focus', 'Malibu', 'Altima', 'Camry', 'Accord', 'Fiesta', 'Impala', 'Rogue'],
    'year': [2018, 2019, 2017, 2020, 2018, 2017, 2019, 2016, 2018, 2019],
    'mileage': [30000, 25000, 50000, 20000, 35000, 45000, 27000, 60000, 40000, 30000],
    'condition': ['Good', 'Excellent', 'Fair', 'Excellent', 'Good', 'Good', 'Excellent', 'Fair', 'Good', 'Excellent'],
    'price': [15000, 18000, 13000, 22000, 16000, 14000, 19000, 12000, 17000, 21000]
})
X = data.drop('price', axis=1)
y = data['price']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['make', 'model', 'condition'])
    ],
    remainder='passthrough'
)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
new_data = pd.DataFrame({
    'make': ['Toyota'],
    'model': ['Camry'],
    'year': [2021],
    'mileage': [20000],
    'condition': ['Excellent']
})
predictions = pipeline.predict(new_data)
print(predictions)
