import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
data = pd.DataFrame({
    'location': ['Downtown', 'Suburb', 'Downtown', 'Suburb', 'Rural', 'Downtown', 'Suburb', 'Rural', 'Downtown', 'Suburb'],
    'size': [1500, 2000, 850, 1750, 2400, 1300, 1800, 3000, 900, 1500],
    'bedrooms': [3, 4, 2, 3, 4, 3, 3, 5, 2, 3],
    'bathrooms': [2, 3, 1, 2, 3, 2, 2, 4, 1, 2],
    'year_built': [2005, 2010, 1980, 2000, 2015, 1995, 2005, 2018, 1970, 1990],
    'price': [350000, 450000, 220000, 400000, 480000, 320000, 410000, 550000, 200000, 360000]
})
X = data.drop('price', axis=1)
y = data['price']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['location'])
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
    'location': ['Downtown'],
    'size': [1500],
    'bedrooms': [3],
    'bathrooms': [2],
    'year_built': [2005]
})
predictions = pipeline.predict(new_data)
print(predictions)
