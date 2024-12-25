import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, roc_curve, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Paths to the CSV files
csv_files = {
    'calc_train': 'csv/calc_case_description_train_set.csv',
    'calc_test': 'csv/calc_case_description_test_set.csv',
    'mass_train': 'csv/mass_case_description_train_set.csv',
    'mass_test': 'csv/mass_case_description_test_set.csv'
}

print("Loading CSV files...")

# Load and process CSV files
def load_and_process_data(csv_files):
    combined_data = []
    for key, file_path in csv_files.items():
        print(f"Processing {key} dataset: {file_path}")

        # Read CSV file
        data = pd.read_csv(file_path)

        # Encode pathology column to binary (0 = Benign, 1 = Malignant)
        data['Label'] = data['pathology'].apply(lambda x: 1 if x.strip().lower() == 'malignant' else 0)

        # Add a column indicating train or test for later splitting
        data['is_train'] = 'train' in key.lower()

        combined_data.append(data)

    # Combine all datasets
    return pd.concat(combined_data, axis=0)

# Combine all data
combined_data = load_and_process_data(csv_files)

print("CSV files loaded successfully!")
print(f"Combined data shape: {combined_data.shape}")

# Columns to exclude
features_to_exclude = ['patient_id', 'pathology', 'image file path', 'cropped image file path', 'ROI mask file path', 'Label', 'is_train']

# Convert non-numerical columns to numerical using LabelEncoder
categorical_cols = combined_data.select_dtypes(include=['object']).columns.difference(features_to_exclude)
label_encoders = {}

print("\nConverting categorical features to numerical values...")
for col in categorical_cols:
    print(f"Encoding column: {col}")
    label_encoders[col] = LabelEncoder()
    combined_data[col] = label_encoders[col].fit_transform(combined_data[col])

# Split data into train and test
train_data = combined_data[combined_data['is_train']].drop(columns=['is_train'], errors='ignore')
test_data = combined_data[~combined_data['is_train']].drop(columns=['is_train'], errors='ignore')

# Separate features and labels
X_train = train_data.drop(columns=features_to_exclude, errors='ignore')
y_train = train_data['Label']
X_test = test_data.drop(columns=features_to_exclude, errors='ignore')
y_test = test_data['Label']

# Handle missing values using SimpleImputer
print("Handling missing values...")
imputer = SimpleImputer(strategy="mean")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

# Train an XGBoost model
print("Training XGBoost model...")
model = XGBClassifier(
    random_state=42,
    n_estimators=100,  # Number of trees
    max_depth=6,  # Limit the depth of trees
    learning_rate=0.1,  # Step size shrinkage
    subsample=0.8,  # Subsample ratio of the training instances
    colsample_bytree=0.8,  # Subsample ratio of columns when constructing trees
    eval_metric="logloss",  # Evaluation metric
)

# Evaluate model using cross-validation
print("Performing cross-validation...")
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
print(f"Cross-Validation F1 Scores: {cv_scores}")
print(f"Mean Cross-Validation F1 Score: {cv_scores.mean():.4f}")

# Fit the model on the entire training set
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

print(f"\nXGBoost Model Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC Score: {auc_score:.4f}")
print("\nClassification Report:")
print(pd.DataFrame(report).transpose())

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
plt.title("Feature Importance in XGBoost")
plt.gca().invert_yaxis()
plt.show()

print("Analysis complete.")
