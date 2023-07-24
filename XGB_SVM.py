import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Load your dataset (replace 'data.csv' with the actual filename)
data = pd.read_csv('data.csv')

# Separate features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create SVM and XGBoost classifiers
svm_model = SVC(kernel='linear', C=1.0)
xgb_model = XGBClassifier()

# Create a list of classifiers for cross-validation
classifiers = [svm_model, xgb_model]

# Perform cross-validation with KFold
num_folds = 5
results = []
for clf in classifiers:
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_scores)
    print(f"{clf.__class__.__name__} - Mean Accuracy: {np.mean(cv_scores):.4f}, Std Deviation: {np.std(cv_scores):.4f}")

# Evaluate on the test set
for clf in classifiers:
    clf.fit(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)
    print(f"{clf.__class__.__name__} - Test Accuracy: {test_accuracy:.4f}")

# The data is split into training and testing sets, and the features are standardized using 'StandardScaler'.
# Replace 'data.csv' with the actual filename of your dataset.
