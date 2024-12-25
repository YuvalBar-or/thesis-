import pandas as pd
import json
import os
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, roc_curve, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Paths to CSV and JSON files
train_csv_path = 'csv/mass_case_description_train_set.csv'
test_csv_path = 'csv/mass_case_description_test_set.csv'
json_folder_path = 'Extracted_Feature_JSON'  # Folder containing JSON files

# Load pathology labels from CSV files
def load_pathology_labels(csv_path):
    print(f"Loading pathology labels from: {csv_path}")
    data = pd.read_csv(csv_path)
    pathology_dict = data.set_index('patient_id')['pathology'].str.strip().str.lower().to_dict()
    return pathology_dict

print("Loading pathology labels...")
train_labels = load_pathology_labels(train_csv_path)
test_labels = load_pathology_labels(test_csv_path)
print("Pathology labels loaded successfully!")

# Load JSON files and prepare datasets
def load_data_from_json(json_files, pathology_labels, json_folder):
    data = []
    labels = []
    for json_file in json_files:
        json_path = os.path.join(json_folder, json_file)
        with open(json_path, 'r') as file:
            features = json.load(file)
            patient_id = features['patient_id']

            # Match pathology label using patient_id
            if patient_id in pathology_labels:
                label = 1 if pathology_labels[patient_id] == 'malignant' else 0
                labels.append(label)
                # Remove patient_id from features for training
                features.pop('patient_id', None)
                data.append(features)
    return pd.DataFrame(data), pd.Series(labels)

# Separate JSON files into train and test based on their filenames
print("Separating JSON files into train and test sets...")
json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]
train_json_files = [f for f in json_files if 'train' in f.lower()]
test_json_files = [f for f in json_files if 'test' in f.lower()]

print(f"Found {len(train_json_files)} training JSON files and {len(test_json_files)} testing JSON files.")

# Load training and testing datasets
print("Loading training and testing datasets...")
X_train, y_train = load_data_from_json(train_json_files, train_labels, json_folder_path)
X_test, y_test = load_data_from_json(test_json_files, test_labels, json_folder_path)
print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

# Train an XGBoost model
print("Training XGBoost model...")
xgb_model = xgb.XGBClassifier(
    random_state=42,
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)
print("Model training completed.")

# Make predictions
print("Making predictions...")
y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]

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
importance_types = ['weight', 'gain', 'cover']
for importance_type in importance_types:
    importances = xgb_model.get_booster().get_score(importance_type=importance_type)
    importance_df = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance'])
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print(f"\nTop Features ({importance_type} importance):")
    print(importance_df.head(10))

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title(f"Feature Importance ({importance_type}) in XGBoost")
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
