##############################################################################################################
## ON THIS TEST WE RAN XGBOOST ON THE BIRADS ONLY - EXCLUDING ASSESSMENT (BIRADS) FEATURE ##
##############################################################################################################

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, roc_curve, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Paths to the CSV files
train_file_path = 'csv/mass_case_description_train_set.csv'
test_file_path = 'csv/mass_case_description_test_set.csv'

print("Loading CSV files...")

# Read the CSV files
train_data_csv = pd.read_csv(train_file_path)
test_data_csv = pd.read_csv(test_file_path)

print("CSV files loaded successfully!")

# Convert 'pathology' column to binary (0 = Benign, 1 = Malignant)
train_data_csv['Label'] = train_data_csv['pathology'].apply(lambda x: 1 if x.strip().lower() == 'malignant' else 0)
test_data_csv['Label'] = test_data_csv['pathology'].apply(lambda x: 1 if x.strip().lower() == 'malignant' else 0)

print("Pathology column encoded as Label (0 = Benign, 1 = Malignant).")

# Combine train and test datasets for consistent encoding
combined_data = pd.concat([train_data_csv, test_data_csv], axis=0)

# List of columns to exclude
features_to_exclude = ['patient_id', 'pathology', 'image file path', 'cropped image file path',
                       'ROI mask file path', 'Label', 'assessment']  # Exclude 'assessment' (BIRADS)

# Convert non-numerical columns to numerical using LabelEncoder
categorical_cols = combined_data.select_dtypes(include=['object']).columns.difference(features_to_exclude)
label_encoders = {}

print("\nConverting categorical features to numerical values...")
for col in categorical_cols:
    print(f"Encoding column: {col}")
    label_encoders[col] = LabelEncoder()
    combined_data[col] = label_encoders[col].fit_transform(combined_data[col])

# Split back into train and test datasets
train_data_processed = combined_data.loc[train_data_csv.index]
test_data_processed = combined_data.loc[test_data_csv.index]

# Separate features and target, excluding 'assessment'
X_train = train_data_processed.drop(columns=features_to_exclude, errors='ignore')
y_train = train_data_processed['Label']
X_test = test_data_processed.drop(columns=features_to_exclude, errors='ignore')
y_test = test_data_processed['Label']

print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

# Train an XGBoost model
print("Training XGBoost model...")
model = XGBClassifier(
    random_state=42,
    n_estimators=100,  # Number of boosting rounds
    max_depth=10,  # Maximum depth of a tree
    learning_rate=0.1,  # Step size shrinkage
    scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),  # Handle class imbalance
    eval_metric='logloss'  # Evaluation metric
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

print(f"XGBoost Model Accuracy: {accuracy:.4f}")
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
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance in XGBoost")
plt.gca().invert_yaxis()
plt.show()

print("Analysis complete.")
