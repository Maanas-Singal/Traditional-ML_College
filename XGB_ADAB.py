import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier

# Load your dataset and separate features (X) and target variable (y)
# Replace 'dataset.csv' with your actual dataset file name
dataset = pd.read_csv('dataset.csv')
X = dataset.drop('target_column', axis=1)  # Adjust 'target_column' to your target column name
y = dataset['target_column']

# Initialize XGBoost classifier
xgb_model = XGBClassifier()

# Initialize AdaBoost classifier with XGBoost as the base estimator
adaboost_model = AdaBoostClassifier(base_estimator=xgb_model)

# Set the number of cross-validation folds
n_folds = 5

# Initialize K-Fold cross-validator
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Perform cross-validation with the AdaBoost model
# This will return an array of accuracy scores for each fold
cross_val_scores = cross_val_score(adaboost_model, X, y, cv=kf)

# Print the accuracy scores for each fold
print("Cross-validation accuracy scores:")
print(cross_val_scores)

# Calculate and print the mean accuracy and standard deviation
mean_accuracy = np.mean(cross_val_scores)
std_accuracy = np.std(cross_val_scores)
print("\nMean Accuracy: {:.2f}%".format(mean_accuracy * 100))
print("Standard Deviation: {:.2f}".format(std_accuracy))

# Make sure to adjust the 'dataset.csv' file name and 'target_column' according to your actual dataset.
# Remember to preprocess your data, handle missing values, and possibly tune hyperparameters for better performance before running cross-validation.
