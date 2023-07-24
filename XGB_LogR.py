import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score

# Load your dataset into a pandas DataFrame (assuming it has features and target columns)
# Replace 'data.csv' with your actual dataset file name and appropriate path.
data = pd.read_csv('data.csv')

# Separate features (X) and target (y)
X = data.drop(columns=['target'])
y = data['target']

# Define the parameters for XGBoost
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.1,
    'max_depth': 3,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

# Initialize XGBoost classifier
xgb_classifier = xgb.XGBClassifier(**xgb_params)

# Initialize logistic regression classifier
logreg_classifier = LogisticRegression()

# Cross-validation using XGBoost
# We'll use cross_val_predict to get predicted probabilities for each data point.
# The predicted probabilities will be used as features for logistic regression.
# This is known as "stacking" or "blending" of models.
xgb_cv_pred_prob = cross_val_predict(xgb_classifier, X, y, cv=5, method='predict_proba')[:, 1]

# Train logistic regression on the predicted probabilities from XGBoost
logreg_classifier.fit(xgb_cv_pred_prob.reshape(-1, 1), y)

# Cross-validation using logistic regression with the stacked features
logreg_cv_pred = cross_val_predict(logreg_classifier, xgb_cv_pred_prob.reshape(-1, 1), y, cv=5)

# Calculate accuracy and ROC AUC score of the final model
final_accuracy = accuracy_score(y, logreg_cv_pred)
final_roc_auc = roc_auc_score(y, logreg_cv_pred)

print("Final Model Accuracy:", final_accuracy)
print("Final Model ROC AUC Score:", final_roc_auc)

# Replace 'data.csv' with the actual filename and path of your dataset file.
