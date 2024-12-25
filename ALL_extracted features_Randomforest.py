import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    roc_curve,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

# Paths to CSV and JSON files
csv_files = {
    'calc_train': 'csv/calc_case_description_train_set.csv',
    'calc_test': 'csv/calc_case_description_test_set.csv',
    'mass_train': 'csv/mass_case_description_train_set.csv',
    'mass_test': 'csv/mass_case_description_test_set.csv'
}
json_folder_path = 'Extracted_Feature_JSON'

# Load pathology labels from CSV files
def load_pathology_labels(csv_path):
    print(f"Loading pathology labels from: {csv_path}")
    data = pd.read_csv(csv_path)
    pathology_dict = data.set_index('patient_id')['pathology'].str.strip().str.lower().to_dict()
    return pathology_dict

print("Loading pathology labels...")
pathology_labels = {}
for key, csv_path in csv_files.items():
    pathology_labels[key] = load_pathology_labels(csv_path)
print("Pathology labels loaded successfully!")

# Load JSON files and prepare datasets
def load_data_from_json(json_files, pathology_labels, json_folder):
    data = []
    labels = []
    for json_file in json_files:
        json_path = os.path.join(json_folder, json_file)
        with open(json_path, 'r') as file:
            features = json.load(file)
            patient_id = features.pop('patient_id', None)

            # Match pathology label using patient_id
            if patient_id in pathology_labels:
                label = 1 if pathology_labels[patient_id] == 'malignant' else 0
                labels.append(label)
                data.append(features)
    return pd.DataFrame(data), pd.Series(labels)

# Separate JSON files into train and test based on their filenames
print("Separating JSON files into train and test sets...")
json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]
train_json_files = [f for f in json_files if 'train' in f.lower()]
test_json_files = [f for f in json_files if 'test' in f.lower()]

print(f"Found {len(train_json_files)} training JSON files and {len(test_json_files)} testing JSON files.")

# Load training and testing datasets for both `mass` and `calc`
X_train, y_train = pd.DataFrame(), pd.Series(dtype=int)
X_test, y_test = pd.DataFrame(), pd.Series(dtype=int)

for key in csv_files:
    dataset_type = 'train' if 'train' in key else 'test'
    json_file_subset = train_json_files if dataset_type == 'train' else test_json_files
    pathology_label_key = f"{key.split('_')[0]}_{dataset_type}"

    if pathology_label_key in pathology_labels:
        X, y = load_data_from_json(json_file_subset, pathology_labels[pathology_label_key], json_folder_path)
        if dataset_type == 'train':
            X_train = pd.concat([X_train, X], ignore_index=True)
            y_train = pd.concat([y_train, y], ignore_index=True)
        else:
            X_test = pd.concat([X_test, X], ignore_index=True)
            y_test = pd.concat([y_test, y], ignore_index=True)

print(f"Final Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Train a Random Forest model
print("Training Random Forest model...")
model = RandomForestClassifier(
    random_state=42,
    n_estimators=100,  # Number of trees in the forest
    max_depth=10,  # Limit the depth of trees
    min_samples_split=5,  # Minimum samples required to split a node
    min_samples_leaf=2,  # Minimum samples in leaf nodes
    class_weight="balanced",  # Handle class imbalance
)

model.fit(X_train, y_train)
print("Model training completed.")

# Perform cross-validation
print("Evaluating with cross-validation...")
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
print(f"Cross-Validation F1 Scores: {cv_scores}")
print(f"Mean Cross-Validation F1 Score: {np.mean(cv_scores):.4f}")

# Make predictions
print("Making predictions...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_prob)
report = classification_report(y_test, y_pred, output_dict=True)

print("\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC Score: {auc_score:.4f}")
print("\nClassification Report:")
print(pd.DataFrame(report).transpose())

# Feature Importance Analysis
print("\nAnalyzing feature importance...")
importances = model.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nTop 10 Important Features:")
print(importance_df.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in Random Forest")
plt.gca().invert_yaxis()
plt.show()

# Plot ROC Curve
print("Plotting ROC Curve...")
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# Plot Confusion Matrix
print("Plotting Confusion Matrix...")
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.show()

print("Analysis complete.")
