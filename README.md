# code
codealpha intern


task 1




To create a system that predicts whether a person would survive the sinking of the Titanic, you can use a machine learning model to classify passengers based on features like socio-economic status (class), age, gender, and more. Here's a step-by-step guide to build this classification system using Python and the popular machine learning library, scikit-learn.

Step 1: Data Preparation
First, you'll need the Titanic dataset, which is commonly used for this type of task. You can download it from Kaggle's Titanic dataset.

Step 2: Load and Explore the Data
Load the dataset and explore it to understand its structure and contents.

python

import pandas as pd

# Load the dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Display the first few rows of the dataset
print(train_data.head())
Step 3: Data Preprocessing
Clean and preprocess the data by handling missing values, converting categorical variables to numerical, and scaling the data.

python

from sklearn.preprocessing import LabelEncoder

def preprocess_data(data):
    # Fill missing values
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    
    # Convert categorical variables to numerical
    data['Sex'] = LabelEncoder().fit_transform(data['Sex'])
    data['Embarked'] = LabelEncoder().fit_transform(data['Embarked'])
    
    # Drop unnecessary columns
    data.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
    
    return data

train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)
Step 4: Feature Selection
Select the features that are most likely to influence survival.

python

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X_train = train_data[features]
y_train = train_data['Survived']
X_test = test_data[features]
Step 5: Train a Machine Learning Model
Train a machine learning model, such as a Random Forest Classifier, to predict survival.

python

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the training set
train_predictions = model.predict(X_train)

# Evaluate the model
accuracy = accuracy_score(y_train, train_predictions)
print(f'Training Accuracy: {accuracy:.2f}')
Step 6: Make Predictions on Test Data
Use the trained model to make predictions on the test data.

python

# Predict on the test set
test_predictions = model.predict(X_test)

# Create a submission file
submission = pd.DataFrame({
    'PassengerId': pd.read_csv('test.csv')['PassengerId'],
    'Survived': test_predictions
})

# Save the submission file
submission.to_csv('submission.csv', index=False)
Step 7: Identify Important Features
Determine which features were most influential in predicting survival.

python

importances = model.feature_importances_
feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

print(feature_importance)
Summary
This system preprocesses the Titanic dataset, selects relevant features, trains a Random Forest Classifier, and makes predictions on whether a person would survive based on socio-economic status, age, gender, and other factors. The feature importance analysis helps identify which factors were most critical in determining survival.
