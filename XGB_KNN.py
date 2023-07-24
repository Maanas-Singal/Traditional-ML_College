import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
# Ensure that the target variable is in the 'target_column' column
dataset = pd.read_csv('your_dataset.csv')
target_column = 'target'

# Separate features and target
X = dataset.drop(target_column, axis=1)
y = dataset[target_column]

# Split the data into training and testing sets (not needed for cross-validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Define the XGBoost classifier
xgb_classifier = xgb.XGBClassifier()

# Perform cross-validation using KNN and XGBoost
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Cross-validation using KNN
knn_scores = cross_val_score(knn_classifier, X, y, cv=kf, scoring='accuracy')

# Cross-validation using XGBoost
xgb_scores = cross_val_score(xgb_classifier, X, y, cv=kf, scoring='accuracy')

print(f"KNN Cross-Validation Scores: {knn_scores}")
print(f"KNN Mean Accuracy: {np.mean(knn_scores)}")

print(f"XGBoost Cross-Validation Scores: {xgb_scores}")
print(f"XGBoost Mean Accuracy: {np.mean(xgb_scores)}")

# Now you can fit and evaluate the models on the test set (not needed for cross-validation)
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN Test Accuracy: {accuracy_knn}")

xgb_classifier.fit(X_train, y_train)
y_pred_xgb = xgb_classifier.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Test Accuracy: {accuracy_xgb}")

# Remember that KNN and XGBoost have different hyperparameters that can be tuned to potentially improve their performance.
# Grid search or random search can be used to find optimal hyperparameters.
