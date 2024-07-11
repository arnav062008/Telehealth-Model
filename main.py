#0-FM 1-M
#YES - 1 NO - 0
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import csv

# Load data from the CSV file
csv_file = 'lungs.csv'  # Replace with the path to your CSV file
data = pd.read_csv(csv_file)

# Split the data into features (symptoms) and target (disease labels)
X = data.drop(columns="label")

y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on test data
y_pred = clf.predict(X_test)


# Calculate accuracy (for evaluation)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Confidence Level: {accuracy * 100:.2f}%")

from joblib import dump, load

model_filename = 'lung_cancer_decision_tree_model.joblib'
dump(clf, model_filename)
print(f"Model saved to {model_filename}")


# Predict a disease for new symptoms
def prediction():
    new_symptoms = [[1,60,2,2,2,2,2,1,1,1,2,1,1,1,1]]

    predicted_disease = clf.predict(new_symptoms)
    print(predicted_disease[0])

prediction()



