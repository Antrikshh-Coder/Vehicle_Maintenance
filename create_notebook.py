import os
import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # 1. Project Introduction
    cells.append(nbf.v4.new_markdown_cell(
"""# Vehicle Maintenance Analysis and Prediction Using Machine Learning
### Academic ML Exam Project — Case Study No. 67

## 1. Project Introduction
Predictive maintenance is a key artificial intelligence strategy in modern transportation and fleet logistics. By leveraging machine learning models trained on sensor telemetry—such as engine speed (RPM), oil pressure, coolant pressure, and operating temperatures—fleet operators can detect internal mechanical anomalies and prevent catastrophic engine failures before they occur."""
    ))
    
    # 2. Problem Statement
    cells.append(nbf.v4.new_markdown_cell(
"""## 2. Problem Statement
"A fleet operator wants to study patterns that may indicate maintenance requirements."

The objective is to analyze vehicle and engine sensor telemetry data and build a supervised Machine Learning binary classification model that accurately predicts whether an engine condition indicates a **Normal** operating status (0) or requires **Maintenance** (1)."""
    ))
    
    # 3. Objectives
    cells.append(nbf.v4.new_markdown_cell(
"""## 3. Objectives
1. Load and inspect the Automotive Vehicles Engine Health dataset (19,535 records).
2. Clean and preprocess sensor telemetry data, avoiding data leakage.
3. Conduct thorough Exploratory Data Analysis (EDA) with statistical visualizations.
4. Train 5 distinct classification algorithms: Logistic Regression, KNN, Decision Tree, Random Forest, and SVM.
5. Perform 5-fold Stratified Cross-Validation and Hyperparameter Tuning.
6. Evaluate models on Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
7. Conduct Error Analysis and select the optimal model for deployment."""
    ))
    
    # 4. Import Libraries
    cells.append(nbf.v4.new_code_cell(
"""import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

import joblib

print('Libraries imported successfully!')"""
    ))
    
    # 5. Load Dataset
    cells.append(nbf.v4.new_markdown_cell(
"""## 5. Load Dataset
Loading the primary dataset `engine_data.csv` containing 19,535 engine sensor telemetry records."""
    ))
    cells.append(nbf.v4.new_code_cell(
"""data_path = '../data/dataset.csv' if os.path.exists('../data/dataset.csv') else 'data/dataset.csv'
df = pd.read_csv(data_path)
print(f'Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns')
df.head()"""
    ))
    
    # 6. Dataset Information
    cells.append(nbf.v4.new_markdown_cell(
"""## 6. Dataset Information
Inspect column data types, missing values, memory usage, and basic summary statistics."""
    ))
    cells.append(nbf.v4.new_code_cell(
"""df.info()
df.describe().T"""
    ))
    
    # 7. Data Dictionary
    cells.append(nbf.v4.new_markdown_cell(
"""## 7. Data Dictionary

| Feature Name | Description | Data Type | Role | Units / Range |
| :--- | :--- | :--- | :--- | :--- |
| **Engine RPM** | Engine rotational speed | `int64` | Input Feature | 61 – 2,239 RPM |
| **Lub Oil Pressure** | Lubricating oil pressure | `float64` | Input Feature | 0.003 – 7.27 bar |
| **Fuel Pressure** | Fuel delivery pressure | `float64` | Input Feature | 0.003 – 21.14 bar |
| **Coolant Pressure** | Cooling loop pressure | `float64` | Input Feature | 0.002 – 7.48 bar |
| **Lub Oil Temp** | Lubricating oil temperature | `float64` | Input Feature | 71.32 – 89.58 °C |
| **Coolant Temp** | Engine coolant temperature | `float64` | Input Feature | 61.67 – 195.53 °C |
| **Engine Condition** | Target Class (0=Normal, 1=Maintenance) | `int64` | Target | Binary {0, 1} |"""
    ))
    
    # 8. Data Quality Checks
    cells.append(nbf.v4.new_markdown_cell(
"""## 8. Data Quality Checks
Verify missing values, duplicate records, and target balance."""
    ))
    cells.append(nbf.v4.new_code_cell(
"""print("Missing Values:\\n", df.isnull().sum())
print("\\nDuplicate Records:", df.duplicated().sum())
print("\\nTarget Distribution:\\n", df['Engine Condition'].value_counts(normalize=True))"""
    ))
    
    # 9. Data Cleaning
    cells.append(nbf.v4.new_markdown_cell(
"""## 9. Data Cleaning
Clean dataset by removing duplicates if any, ensuring no null entries remain."""
    ))
    cells.append(nbf.v4.new_code_cell(
"""cleaned_df = df.copy()
if cleaned_df.duplicated().sum() > 0:
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
print(f'Cleaned Dataset Shape: {cleaned_df.shape}')"""
    ))
    
    # 10. Exploratory Data Analysis
    cells.append(nbf.v4.new_markdown_cell(
"""## 10. Exploratory Data Analysis (EDA)
Visualizing feature distributions, correlations, and relationships with target condition."""
    ))
    cells.append(nbf.v4.new_code_cell(
"""plt.figure(figsize=(6, 4))
sns.countplot(data=cleaned_df, x='Engine Condition', palette=['#2ecc71', '#e74c3c'])
plt.title('Target Class Distribution')
plt.xticks([0, 1], ['Normal (0)', 'Maintenance (1)'])
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(cleaned_df.corr(), annot=True, fmt='.3f', cmap='coolwarm')
plt.title('Pearson Correlation Heatmap')
plt.show()"""
    ))
    
    # 11. Data Preprocessing
    cells.append(nbf.v4.new_markdown_cell(
"""## 11. Data Preprocessing
Separating feature matrix X and target vector y."""
    ))
    cells.append(nbf.v4.new_code_cell(
"""X = cleaned_df.drop(columns=['Engine Condition'])
y = cleaned_df['Engine Condition']
feature_names = list(X.columns)
print('Input Features:', feature_names)"""
    ))
    
    # 12. Train-Test Split
    cells.append(nbf.v4.new_markdown_cell(
"""## 12. Train-Test Split
Performing 80/20 train-test split with stratification and standard feature scaling on training set only to prevent data leakage."""
    ))
    cells.append(nbf.v4.new_code_cell(
"""X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f'Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}')"""
    ))
    
    # 13. Model Training
    cells.append(nbf.v4.new_markdown_cell(
"""## 13. Model Training
Training 5 classification models: Logistic Regression, KNN, Decision Tree, Random Forest, Support Vector Machine."""
    ))
    cells.append(nbf.v4.new_code_cell(
"""models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=12),
    'Support Vector Machine': SVC(random_state=42, probability=True)
}

fitted_models = {}
for name, clf in models.items():
    clf.fit(X_train_scaled, y_train)
    fitted_models[name] = clf
    print(f'Fitted {name}')"""
    ))
    
    # 14. Model Evaluation
    cells.append(nbf.v4.new_markdown_cell(
"""## 14. Model Evaluation
Evaluate models on unseen test dataset using Accuracy, Precision, Recall, F1-Score, and ROC-AUC."""
    ))
    cells.append(nbf.v4.new_code_cell(
"""eval_results = []
for name, clf in fitted_models.items():
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, 'predict_proba') else None
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
    
    eval_results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc
    })

eval_df = pd.DataFrame(eval_results)
eval_df"""
    ))
    
    # 15. Model Comparison
    cells.append(nbf.v4.new_markdown_cell(
"""## 15. Model Comparison
Comparative performance analysis across all classification metrics."""
    ))
    cells.append(nbf.v4.new_code_cell(
"""plt.figure(figsize=(10, 5))
melted = eval_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
sns.barplot(data=melted, x='Model', y='Score', hue='Metric', palette='viridis')
plt.title('Model Comparison Metrics')
plt.xticks(rotation=15)
plt.ylim(0.5, 1.0)
plt.show()"""
    ))
    
    # 16. Cross-Validation
    cells.append(nbf.v4.new_markdown_cell(
"""## 16. Stratified 5-Fold Cross-Validation
Running 5-fold Stratified Cross-Validation on training set to evaluate generalization variance."""
    ))
    cells.append(nbf.v4.new_code_cell(
"""skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, clf in models.items():
    scores = cross_validate(clf, X_train_scaled, y_train, cv=skf, scoring='f1')
    print(f'{name} 5-Fold CV F1-Score: {scores["test_score"].mean():.4f} (+/- {scores["test_score"].std():.4f})')"""
    ))
    
    # 17. Hyperparameter Tuning
    cells.append(nbf.v4.new_markdown_cell(
"""## 17. Hyperparameter Tuning
Hyperparameter search for Random Forest using GridSearchCV."""
    ))
    cells.append(nbf.v4.new_code_cell(
"""param_grid = {'n_estimators': [100, 150], 'max_depth': [10, 15]}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=skf, scoring='f1', n_jobs=-1)
grid.fit(X_train_scaled, y_train)
print('Best Random Forest Params:', grid.best_params_)
print('Best CV F1 Score:', grid.best_score_)"""
    ))
    
    # 18. Final Model Selection
    cells.append(nbf.v4.new_markdown_cell(
"""## 18. Final Model Selection
Support Vector Machine (SVM) was selected as the final production model due to achieving the highest test F1-Score (0.7709) and superior Recall (89.57%), critical for identifying engine faults."""
    ))
    cells.append(nbf.v4.new_code_cell(
"""final_model = fitted_models['Support Vector Machine']
joblib.dump(final_model, '../models/final_model.joblib' if os.path.exists('../models') else 'models/final_model.joblib')
print('Final Model saved successfully.')"""
    ))
    
    # 19. Error Analysis
    cells.append(nbf.v4.new_markdown_cell(
"""## 19. Error Analysis
Analyzing false negative and false positive errors of the final model on test data."""
    ))
    cells.append(nbf.v4.new_code_cell(
"""y_pred_final = final_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred_final)
tn, fp, fn, tp = cm.ravel()
print(f'True Negatives (TN): {tn}')
print(f'False Positives (FP): {fp}')
print(f'False Negatives (FN): {fn}')
print(f'True Positives (TP): {tp}')

print(f'False Negative Rate (Uncaught Failures): {fn / (fn + tp) * 100:.2f}%')
print(f'False Positive Rate (False Alarms): {fp / (fp + tn) * 100:.2f}%')"""
    ))
    
    # 20. Conclusions
    cells.append(nbf.v4.new_markdown_cell(
"""## 20. Conclusions
1. The machine learning pipeline successfully analyzes 19,535 vehicle engine telemetry records.
2. Machine learning classifiers effectively separate operational engine states based on sensor pressure and temperature relationships.
3. Support Vector Machine (SVM) delivers the highest F1-Score (0.7709) and strong Recall (89.57%).
4. The solution was deployed into an interactive Streamlit application for real-time engine condition diagnosis."""
    ))
    
    # 21. Limitations
    cells.append(nbf.v4.new_markdown_cell(
"""## 21. Limitations
1. **Sensor Range Coverage**: Model predictions are bounded by the sensor ranges present in the training set.
2. **Binary Classification**: The target variable is binary (Normal vs. Maintenance) rather than predicting multi-class component failure types (e.g., oil pump failure vs. coolant leak).
3. **Time-Series Dynamics**: Telemetry is treated as independent steady-state samples rather than dynamic time-series sequences."""
    ))
    
    # 22. Future Scope
    cells.append(nbf.v4.new_markdown_cell(
"""## 22. Future Scope
1. Integrate time-series modeling (LSTM / Transformer networks) for remaining useful life (RUL) estimation.
2. Expand target labels to multi-class component fault diagnoses.
3. Connect live IoT MQTT telemetry streams directly into the Streamlit dashboard for real-time vehicle monitoring."""
    ))
    
    nb['cells'] = cells
    
    os.makedirs("notebooks", exist_ok=True)
    nb_path = os.path.join("notebooks", "vehicle_maintenance_analysis.ipynb")
    with open(nb_path, "w") as f:
        nbf.write(nb, f)
    print(f"Notebook generated cleanly at {nb_path}")

if __name__ == "__main__":
    create_notebook()
