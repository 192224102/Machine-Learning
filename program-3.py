import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'AirTemp': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'High'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Strong'],
    'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
    'Forecast': ['Same', 'Same', 'Change', 'Change'],
    'EnjoySport': ['Yes', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)
X = df.iloc[:, :-1] 
y = df.iloc[:, -1]
X_encoded = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=X_encoded.columns, class_names=clf.classes_, filled=True)
plt.show()
new_sample = {
    'Sky_Sunny': 1,
    'Sky_Rainy': 0,
    'AirTemp_Warm': 1,
    'AirTemp_Cold': 0,
    'Humidity_Normal': 0,
    'Humidity_High': 1,
    'Wind_Strong': 1,
    'Water_Warm': 1,
    'Water_Cool': 0,
    'Forecast_Same': 0,
    'Forecast_Change': 1
}
new_sample_df = pd.DataFrame([new_sample], columns=X_encoded.columns)
prediction = clf.predict(new_sample_df)
print("Prediction for new sample:", prediction[0])
