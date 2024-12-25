import pandas as pd
import json
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, roc_curve, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Paths to CSV files and JSON folder
csv_files = {
    'mass': {
        'train': 'csv/mass_case_description_train_set.csv',
        'test': 'csv/mass_case_description_test_set.csv'
    },
    'calc': {
        'train': 'csv/calc_case_description_train_set.csv',
        'test': 'csv/calc_case_description_test_set.csv'
    }
}
json_folder_path = 'BIRADS_JSON_with'  # Folder containing JSON files

# Combined datasets
combined_train_data = []
combined_test_data = []

# Process each case type (mass and calc)
for case_type in csv_files:
    print(f"Processing {case_type} cases...\n")

    # Load CSV files
    print("Loading CSV files...")
    train_csv_path = csv_files[case_type]['train']
    test_csv_path = csv_files[case_type]['test']
    train_data_csv = pd.read_csv(train_csv_path)
    test_data_csv = pd.read_csv(test_csv_path)
    print("CSV files loaded successfully!")

    # Load JSON files and filter for the specific case type
    print(f"Loading JSON files for {case_type} cases only...")
    json_data = []
    for json_file in os.listdir(json_folder_path):
        if json_file.endswith('.json') and case_type in json_file.lower():
            with open(os.path.join(json_folder_path, json_file), 'r') as file:
                json_data.append(json.load(file))
    json_df = pd.DataFrame(json_data)
    print(f"Loaded {len(json_df)} JSON records related to {case_type} cases.")

    # Merge JSON and CSV data on patient_id
    print("Merging JSON and CSV data...")
    train_data = train_data_csv.merge(json_df, on='patient_id', how='inner')
    test_data = test_data_csv.merge(json_df, on='patient_id', how='inner')
    print("Data merged successfully!")

    # Append to combined datasets
    combined_train_data.append(train_data)
    combined_test_data.append(test_data)

# Combine all train and test data
train_data = pd.concat(combined_train_data, ignore_index=True)
test_data = pd.concat(combined_test_data, ignore_index=True)

# Encode pathology column as binary (0 = Benign, 1 = Malignant)
print("Encoding pathology as binary label...")
train_data['Label'] = train_data['pathology'].apply(lambda x: 1 if x.strip().lower() == 'malignant' else 0)
test_data['Label'] = test_data['pathology'].apply(lambda x: 1 if x.strip().lower() == 'malignant' else 0)

# Exclude unnecessary columns
columns_to_exclude = ['patient_id', 'pathology', 'image file path', 'cropped image file path', 'ROI mask file path']
X_train = train_data.drop(columns=columns_to_exclude + ['Label'], errors='ignore')
y_train = train_data['Label']
X_test = test_data.drop(columns=columns_to_exclude + ['Label'], errors='ignore')
y_test = test_data['Label']

# Handle missing values
print("Handling missing values...")
num_imputer = SimpleImputer(strategy="mean")
cat_imputer = SimpleImputer(strategy="most_frequent")

# Identify numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=["float64", "int64"]).columns
categorical_cols = X_train.select_dtypes(include=["object"]).columns

if not categorical_cols.empty:
    X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
    X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

if not numerical_cols.empty:
    X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

print("Missing values handled.")

# Convert categorical features to numerical using LabelEncoder
label_encoders = {}
for col in categorical_cols:
    print(f"Encoding column: {col}")
    label_encoders[col] = LabelEncoder()
    X_train[col] = label_encoders[col].fit_transform(X_train[col].astype(str))
    X_test[col] = X_test[col].apply(lambda x: label_encoders[col].transform([x])[0]
                                    if x in label_encoders[col].classes_ else -1)

print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

# Train a Decision Tree model
print("Training Decision Tree model...")
model = DecisionTreeClassifier(
    random_state=42,
    max_depth=10,  # Maximum depth of the tree
    min_samples_split=10,  # Minimum samples required to split a node
    min_samples_leaf=5  # Minimum samples required in a leaf node
)

model.fit(X_train, y_train)
print("Model training completed.")

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
plt.title("Feature Importance in Decision Tree")
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
