import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Creating a synthetic bank loan prediction dataset using pandas DataFrame
data = {
    'income': [2000, 4000, 6000, 8000, 10000],
    'credit_score': [600, 700, 750, 800, 850],
    'loan_approved': ['no', 'no', 'yes', 'yes', 'yes']
}
df = pd.DataFrame(data)
# Convert categorical target 'loan_approved' to numerical labels
le = LabelEncoder()
df['loan_approved'] = le.fit_transform(df['loan_approved'])

# Splitting the data into features (X) and target (y)
X = df.drop('loan_approved', axis=1)
y = df['loan_approved']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=44)

# Initializing the Naive Bayes classifier (Gaussian)
model = GaussianNB()

# Training the model
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)
# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Naive Bayes classifier for bank loan prediction: {accuracy}")
