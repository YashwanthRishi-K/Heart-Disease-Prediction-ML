# ============================================================
# HEART DISEASE PREDICTION USING MACHINE LEARNING
# UCI Heart Disease Dataset
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import zipfile
import io
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

print("Libraries imported successfully!")


# ============================================================
# 2. LOAD DATASET FROM UCI
# ============================================================

url = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"

# Download dataset
response = requests.get(url)
response.raise_for_status()

# Open ZIP file
with zipfile.ZipFile(io.BytesIO(response.content)) as z:

    print("\nFiles available in UCI dataset:")
    print(z.namelist())

    # Load Cleveland dataset
    with z.open("processed.cleveland.data") as file:

        column_names = [
            'age',
            'sex',
            'cp',
            'trestbps',
            'chol',
            'fbs',
            'restecg',
            'thalach',
            'exang',
            'oldpeak',
            'slope',
            'ca',
            'thal',
            'target'
        ]

        data = pd.read_csv(
            file,
            names=column_names
        )

print("\nFirst 5 rows of the dataset:")
display(data.head())

print("\nDataset shape:")
print(data.shape)


# ============================================================
# 3. DATA PREPROCESSING
# ============================================================

# Replace '?' values with NaN
data.replace('?', np.nan, inplace=True)

# Convert all columns to numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Remove rows containing missing values
data.dropna(inplace=True)

# Convert target into binary classification
# 0 = No heart disease
# 1 = Heart disease

data['target'] = data['target'].apply(
    lambda x: 1 if x > 0 else 0
)

print("\nMissing values after cleaning:")
print(data.isnull().sum())

print("\nDataset shape after cleaning:")
print(data.shape)

print("\nTarget distribution:")
print(data['target'].value_counts())


# ============================================================
# 4. DESCRIPTIVE STATISTICS
# ============================================================

print("\nDescriptive Statistics:")
display(data.describe())


# ============================================================
# 5. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(12, 8))

sns.heatmap(
    data.corr(),
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title(
    'Correlation Heatmap - Heart Disease Dataset',
    fontsize=16
)

plt.show()


# ============================================================
# 6. SEPARATE FEATURES AND TARGET
# ============================================================

X = data.drop('target', axis=1)
y = data['target']

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("0 = No Heart Disease")
print("1 = Heart Disease")


# ============================================================
# 7. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)


# ============================================================
# 8. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed successfully!")


# ============================================================
# 9. LOGISTIC REGRESSION
# ============================================================

log_reg_model = LogisticRegression(
    random_state=42
)

log_reg_model.fit(
    X_train_scaled,
    y_train
)

# Predictions
log_reg_preds = log_reg_model.predict(
    X_test_scaled
)

# Evaluation
print("\n")
print("=" * 60)
print("LOGISTIC REGRESSION MODEL PERFORMANCE")
print("=" * 60)

print("\nAccuracy Score:")
print(accuracy_score(y_test, log_reg_preds))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, log_reg_preds))

print("\nClassification Report:")
print(classification_report(y_test, log_reg_preds))


# ============================================================
# 10. RANDOM FOREST
# ============================================================

rf_model = RandomForestClassifier(
    random_state=42
)

rf_model.fit(
    X_train_scaled,
    y_train
)

# Predictions
rf_preds = rf_model.predict(
    X_test_scaled
)

# Evaluation
print("\n")
print("=" * 60)
print("RANDOM FOREST MODEL PERFORMANCE")
print("=" * 60)

print("\nAccuracy Score:")
print(accuracy_score(y_test, rf_preds))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_preds))

print("\nClassification Report:")
print(classification_report(y_test, rf_preds))


# ============================================================
# 11. HYPERPARAMETER TUNING
# ============================================================

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring='accuracy'
)

print("\n")
print("=" * 60)
print("STARTING GRID SEARCH")
print("=" * 60)

grid_search.fit(
    X_train_scaled,
    y_train
)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(grid_search.best_score_)


# ============================================================
# 12. TUNED RANDOM FOREST
# ============================================================

best_rf_model = grid_search.best_estimator_

# Predictions
rf_tuned_preds = best_rf_model.predict(
    X_test_scaled
)

# Evaluation
print("\n")
print("=" * 60)
print("TUNED RANDOM FOREST MODEL PERFORMANCE")
print("=" * 60)

print("\nAccuracy Score:")
print(accuracy_score(y_test, rf_tuned_preds))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_tuned_preds))

print("\nClassification Report:")
print(classification_report(y_test, rf_tuned_preds))


# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

feature_importances = (
    best_rf_model.feature_importances_
)

features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

features_df = features_df.sort_values(
    by='Importance',
    ascending=False
)

print("\n")
print("=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

display(features_df)


# Feature Importance Graph
plt.figure(figsize=(12, 8))

sns.barplot(
    x='Importance',
    y='Feature',
    data=features_df
)

plt.title(
    'Feature Importance for Heart Disease Prediction',
    fontsize=16
)

plt.xlabel('Importance')
plt.ylabel('Feature')

plt.show()


# ============================================================
# 14. FINAL MODEL
# ============================================================

final_model = best_rf_model

# Final predictions
final_preds = final_model.predict(
    X_test_scaled
)

# Final evaluation
print("\n")
print("=" * 60)
print("FINAL MODEL PERFORMANCE")
print("=" * 60)

print("\nFinal Model: Tuned Random Forest")

print("\nAccuracy Score:")
print(accuracy_score(y_test, final_preds))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, final_preds))

print("\nClassification Report:")
print(classification_report(y_test, final_preds))


# ============================================================
# 15. SAVE TRAINED MODEL
# ============================================================

with open(
    "heart_disease_prediction_model.pkl",
    "wb"
) as model_file:

    pickle.dump(
        final_model,
        model_file
    )

print("\nHeart disease prediction model saved successfully!")


# ============================================================
# 16. SAVE SCALER
# ============================================================

with open(
    "heart_disease_scaler.pkl",
    "wb"
) as scaler_file:

    pickle.dump(
        scaler,
        scaler_file
    )

print("Scaler saved successfully!")


# ============================================================
# 17. DISPLAY FINAL RESULT
# ============================================================

final_accuracy = accuracy_score(
    y_test,
    final_preds
)

print("\n")
print("=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    f"\nFinal Model Accuracy: "
    f"{final_accuracy * 100:.2f}%"
)

print("\nSaved files:")
print("1. heart_disease_prediction_model.pkl")
print("2. heart_disease_scaler.pkl")

Add main.py file
