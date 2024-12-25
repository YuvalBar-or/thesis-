import pandas as pd
import json
import os
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Paths to CSV and JSON files
train_csv_paths = ['csv/mass_case_description_train_set.csv', 'csv/calc_case_description_train_set.csv']
test_csv_paths = ['csv/mass_case_description_test_set.csv', 'csv/calc_case_description_test_set.csv']
json_folder_path = 'BIRADS_JSON_with'  # Folder containing JSON files

# Load CSV files for both mass and calc
print("Loading CSV files...")
train_data_csv = pd.concat([pd.read_csv(path) for path in train_csv_paths], ignore_index=True)
test_data_csv = pd.concat([pd.read_csv(path) for path in test_csv_paths], ignore_index=True)
print(f"Combined training data shape: {train_data_csv.shape}")
print(f"Combined test data shape: {test_data_csv.shape}")

# Load JSON files and combine into a DataFrame
print("Loading JSON files for all cases (mass and calc)...")
json_data = []
for json_file in os.listdir(json_folder_path):
    if json_file.endswith('.json') and ('mass' in json_file.lower() or 'calc' in json_file.lower()):
        with open(os.path.join(json_folder_path, json_file), 'r') as file:
            json_data.append(json.load(file))
json_df = pd.DataFrame(json_data)
print(f"Loaded {len(json_df)} JSON records related to mass and calc cases.")

# Merge JSON and CSV data on patient_id
print("Merging JSON and CSV data...")
train_data = train_data_csv.merge(json_df, on='patient_id', how='inner')
test_data = test_data_csv.merge(json_df, on='patient_id', how='inner')

# Combine duplicate columns (e.g., assessment_x and assessment_y)
for col in train_data.columns:
    if col.endswith('_x') and col.replace('_x', '_y') in train_data.columns:
        # Resolve duplicate columns (prefer non-NaN values from either column)
        combined_col = col.replace('_x', '')
        train_data[combined_col] = train_data[col].combine_first(train_data[col.replace('_x', '_y')])
        test_data[combined_col] = test_data[col].combine_first(test_data[col.replace('_x', '_y')])
        train_data.drop(columns=[col, col.replace('_x', '_y')], inplace=True)
        test_data.drop(columns=[col, col.replace('_x', '_y')], inplace=True)

print("Merged duplicate columns successfully!")

# Encode pathology column as binary (0 = Benign, 1 = Malignant)
print("Encoding pathology as binary label...")
train_data['Label'] = train_data['pathology'].apply(lambda x: 1 if x.strip().lower() == 'malignant' else 0)
test_data['Label'] = test_data['pathology'].apply(lambda x: 1 if x.strip().lower() == 'malignant' else 0)

# Exclude unnecessary columns
columns_to_exclude = ['patient_id', 'pathology', 'image file path', 'cropped image file path', 'ROI mask file path']
features = train_data.drop(columns=columns_to_exclude + ['Label'], errors='ignore').columns

# Iterate through each feature and exclude it in the analysis
for feature_to_ignore in features:
    print(f"\nIgnoring feature: {feature_to_ignore}")

    # Prepare the training and testing data
    X_train = train_data.drop(columns=columns_to_exclude + ['Label', feature_to_ignore], errors='ignore')
    y_train = train_data['Label']
    X_test = test_data.drop(columns=columns_to_exclude + ['Label', feature_to_ignore], errors='ignore')
    y_test = test_data['Label']

    # Handle missing values
    num_imputer = SimpleImputer(strategy="mean")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    numerical_cols = X_train.select_dtypes(include=["float64", "int64"]).columns
    categorical_cols = X_train.select_dtypes(include=["object"]).columns

    if not categorical_cols.empty:
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

    if not numerical_cols.empty:
        X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

    # Encode categorical features
    label_encoders = {}
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        X_train[col] = label_encoders[col].fit_transform(X_train[col].astype(str))
        X_test[col] = X_test[col].apply(lambda x: label_encoders[col].transform([x])[0]
                                        if x in label_encoders[col].classes_ else -1)

    # Train an XGBoost model
    model = XGBClassifier(
        random_state=42,
        n_estimators=100,  # Number of trees
        max_depth=6,  # Maximum depth of trees
        learning_rate=0.1,  # Learning rate
        subsample=0.8,  # Subsample ratio of the training instances
        colsample_bytree=0.8,  # Subsample ratio of columns when constructing each tree
        eval_metric="logloss"  # Default evaluation metric
    )
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Print Accuracy, AUC, and Classification Report
    print(f"Accuracy for iteration excluding {feature_to_ignore}: {accuracy:.4f}")
    print(f"AUC for iteration excluding {feature_to_ignore}: {auc_score:.4f}")
    print("\nClassification Report:")
    print(pd.DataFrame(report).transpose())

    # Feature Importance Analysis
    importances = model.feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Print Top 10 Important Features
    print("\nTop 10 Important Features:")
    print(importance_df.head(10))
    print(f"Iteration with {feature_to_ignore} excluded complete.\n")
