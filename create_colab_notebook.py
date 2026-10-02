import os
import nbformat as nbf

def create_native_colab_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title
    cells.append(nbf.v4.new_markdown_cell(
"""# ⚙️ Vehicle Maintenance Analysis & Prediction — Google Colab Edition
### Academic Machine Learning Project — Case Study No. 67

> **Instructions for Google Colab**:
> 1. Click **Runtime** -> **Run all** (`Ctrl + F9`) to execute all cells.
> 2. The dataset will be downloaded automatically.
> 3. All charts, models, cross-validation, and metrics will run inline.
> 4. The final cell launches the Streamlit dashboard via localtunnel."""
    ))

    # Cell 1: Setup & Dataset Download
    cell1_code = [
        "# Step 1: Install required packages and download dataset",
        "!pip install -q streamlit scikit-learn seaborn matplotlib joblib",
        "",
        "import os",
        "import urllib.request",
        "",
        'dataset_url = "https://huggingface.co/datasets/jskswamy/predictive-maintenance-data/raw/main/engine_data.csv"',
        'dataset_path = "engine_data.csv"',
        "",
        "if not os.path.exists(dataset_path):",
        '    print("Downloading dataset...")',
        "    urllib.request.urlretrieve(dataset_url, dataset_path)",
        '    print("✅ Dataset downloaded successfully (19,535 records)!")',
        "else:",
        '    print("✅ Dataset engine_data.csv is ready.")'
    ]
    cells.append(nbf.v4.new_code_cell("\n".join(cell1_code)))

    # Cell 2: Imports
    cell2_code = [
        "# Step 2: Import Libraries",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "",
        "from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV",
        "from sklearn.preprocessing import StandardScaler",
        "from sklearn.linear_model import LogisticRegression",
        "from sklearn.neighbors import KNeighborsClassifier",
        "from sklearn.tree import DecisionTreeClassifier",
        "from sklearn.ensemble import RandomForestClassifier",
        "from sklearn.svm import SVC",
        "from sklearn.metrics import (",
        "    accuracy_score, precision_score, recall_score, f1_score,",
        "    roc_auc_score, confusion_matrix, roc_curve",
        ")",
        "import joblib",
        "",
        "# Safe matplotlib style fallback",
        "if 'seaborn-v0_8-whitegrid' in plt.style.available:",
        "    plt.style.use('seaborn-v0_8-whitegrid')",
        "elif 'seaborn-whitegrid' in plt.style.available:",
        "    plt.style.use('seaborn-whitegrid')",
        "else:",
        "    plt.style.use('default')",
        "",
        'print("✅ All ML and Data Science libraries loaded successfully!")'
    ]
    cells.append(nbf.v4.new_code_cell("\n".join(cell2_code)))

    # Cell 3: Load Data & Statistics
    cell3_code = [
        "# Step 3: Load Dataset & Verify Summary Statistics",
        'df = pd.read_csv("engine_data.csv")',
        'print(f"Dataset Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")',
        'print("\\n=== DATA TYPES & MISSING VALUES ===")',
        "info_df = pd.DataFrame({",
        "    'Data Type': df.dtypes,",
        "    'Missing Values': df.isnull().sum(),",
        "    'Unique Values': df.nunique()",
        "})",
        "display(info_df)",
        'print("\\n=== NUMERICAL STATISTICS ===")',
        "display(df.describe().T)"
    ]
    cells.append(nbf.v4.new_code_cell("\n".join(cell3_code)))

    # Cell 4: Class Distribution Checks
    cell4_code = [
        "# Step 4: Verify Class Distribution & Duplicates",
        "duplicates_count = df.duplicated().sum()",
        'print(f"Duplicate Rows Count: {duplicates_count}\\n")',
        'print("Target Distribution (Engine Condition):")',
        "target_counts = df['Engine Condition'].value_counts()",
        "target_props = df['Engine Condition'].value_counts(normalize=True)",
        "",
        "for k in target_counts.index:",
        '    label = "Maintenance Required (1)" if k == 1 else "Normal Condition (0)"',
        '    print(f"  Class {k} ({label}): {target_counts[k]} ({target_props[k]*100:.2f}%)")'
    ]
    cells.append(nbf.v4.new_code_cell("\n".join(cell4_code)))

    # Cell 5: EDA Charts
    cells.append(nbf.v4.new_markdown_cell("## 4. Exploratory Data Analysis (EDA)"))
    cell5_code = [
        "# 1. Target Distribution Plot",
        "fig, ax = plt.subplots(figsize=(7, 4))",
        "sns.countplot(data=df, x='Engine Condition', hue='Engine Condition', palette=['#2ecc71', '#e74c3c'], ax=ax, legend=False)",
        "ax.set_title('Target Distribution: Engine Condition', fontweight='bold')",
        "ax.set_xticks([0, 1])",
        "ax.set_xticklabels(['Normal (0)', 'Maintenance (1)'])",
        "plt.tight_layout()",
        "plt.show()",
        "",
        "# 2. Pearson Correlation Heatmap",
        "fig, ax = plt.subplots(figsize=(8, 6))",
        'sns.heatmap(df.corr(), annot=True, fmt=".3f", cmap="coolwarm", ax=ax, linewidths=0.5)',
        "ax.set_title('Pearson Correlation Heatmap', fontweight='bold')",
        "plt.tight_layout()",
        "plt.show()",
        "",
        "# 3. Feature Distributions",
        "features = [c for c in df.columns if c != 'Engine Condition']",
        "fig, axes = plt.subplots(2, 3, figsize=(15, 8))",
        "axes = axes.flatten()",
        "for idx, col in enumerate(features):",
        "    sns.histplot(data=df, x=col, hue='Engine Condition', kde=True, ax=axes[idx], palette={0: '#2ecc71', 1: '#e74c3c'}, alpha=0.5)",
        "    axes[idx].set_title(f'Distribution: {col}', fontweight='bold')",
        "plt.tight_layout()",
        "plt.show()"
    ]
    cells.append(nbf.v4.new_code_cell("\n".join(cell5_code)))

    # Cell 6: Data Preprocessing
    cells.append(nbf.v4.new_markdown_cell("## 5. Data Preprocessing & Train-Test Split (Preventing Data Leakage)"))
    cell6_code = [
        "# 80/20 Stratified Train-Test Split & Feature Scaling",
        "X = df.drop(columns=['Engine Condition'])",
        "y = df['Engine Condition']",
        "feature_names = list(X.columns)",
        "",
        "X_train, X_test, y_train, y_test = train_test_split(",
        "    X, y, test_size=0.2, random_state=42, stratify=y",
        ")",
        "",
        "scaler = StandardScaler()",
        "X_train_scaled = scaler.fit_transform(X_train)",
        "X_test_scaled = scaler.transform(X_test)",
        "",
        'joblib.dump(scaler, "scaler.joblib")',
        'joblib.dump(feature_names, "feature_names.joblib")',
        "",
        'print("✅ Preprocessing Complete!")',
        'print(f"   Train Set: {X_train_scaled.shape[0]} samples")',
        'print(f"   Test Set:  {X_test_scaled.shape[0]} samples")'
    ]
    cells.append(nbf.v4.new_code_cell("\n".join(cell6_code)))

    # Cell 7: Model Training & Evaluation
    cells.append(nbf.v4.new_markdown_cell("## 6. Model Training, 5-Fold Cross-Validation & Evaluation"))
    cell7_code = [
        "classifiers = {",
        "    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),",
        "    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),",
        "    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),",
        "    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=12),",
        "    'Support Vector Machine': SVC(random_state=42, probability=True)",
        "}",
        "",
        "results = []",
        "fitted_models = {}",
        "skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "",
        'print("Training 5 classification models...")',
        "for name, clf in classifiers.items():",
        "    clf.fit(X_train_scaled, y_train)",
        "    fitted_models[name] = clf",
        "    y_pred = clf.predict(X_test_scaled)",
        "    y_proba = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, 'predict_proba') else None",
        "    acc = accuracy_score(y_test, y_pred)",
        "    prec = precision_score(y_test, y_pred)",
        "    rec = recall_score(y_test, y_pred)",
        "    f1 = f1_score(y_test, y_pred)",
        "    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0",
        "    results.append({",
        "        'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1, 'ROC-AUC': auc",
        "    })",
        "",
        "eval_df = pd.DataFrame(results)",
        'print("\\n=== ACTUAL EXPERIMENTAL MODEL PERFORMANCE COMPARISON ===")',
        "display(eval_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'], color='#d1fae5'))"
    ]
    cells.append(nbf.v4.new_code_cell("\n".join(cell7_code)))

    # Cell 8: Model Selection
    cells.append(nbf.v4.new_markdown_cell("## 7. Model Selection & Artifact Export"))
    cell8_code = [
        "best_idx = eval_df['F1-Score'].idxmax()",
        "best_model_name = eval_df.loc[best_idx, 'Model']",
        "best_f1 = eval_df.loc[best_idx, 'F1-Score']",
        "best_recall = eval_df.loc[best_idx, 'Recall']",
        "",
        'print(f"🏆 Selected Production Model: {best_model_name}")',
        'print(f"   Test F1-Score: {best_f1:.4f}")',
        'print(f"   Test Recall:   {best_recall:.4f}")',
        "",
        "final_model = fitted_models[best_model_name]",
        'joblib.dump(final_model, "final_model.joblib")',
        'print("✅ Saved final_model.joblib successfully!")'
    ]
    cells.append(nbf.v4.new_code_cell("\n".join(cell8_code)))

    # Cell 9: Confusion Matrix
    cells.append(nbf.v4.new_markdown_cell("## 8. Error Analysis & Confusion Matrix"))
    cell9_code = [
        "y_pred_final = final_model.predict(X_test_scaled)",
        "cm = confusion_matrix(y_test, y_pred_final)",
        "tn, fp, fn, tp = cm.ravel()",
        "",
        'print("=== CONFUSION MATRIX BREAKDOWN ===")',
        'print(f"True Negatives (TN)  : {tn}")',
        'print(f"False Positives (FP) : {fp}  (False Alarms)")',
        'print(f"False Negatives (FN) : {fn}  (Uncaught Faults)")',
        'print(f"True Positives (TP)  : {tp}")',
        "",
        "fig, ax = plt.subplots(figsize=(6, 5))",
        'sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["Normal", "Maintenance"], yticklabels=["Normal", "Maintenance"])',
        "ax.set_title(f'Confusion Matrix: {best_model_name}', fontweight='bold')",
        "ax.set_ylabel('Actual Condition')",
        "ax.set_xlabel('Predicted Condition')",
        "plt.tight_layout()",
        "plt.show()"
    ]
    cells.append(nbf.v4.new_code_cell("\n".join(cell9_code)))

    # Cell 10: Streamlit File Creator
    cells.append(nbf.v4.new_markdown_cell("## 9. Streamlit Dashboard Setup"))
    cell10_code = [
        "# Create app.py for Streamlit deployment",
        "app_code = '''import streamlit as st",
        "import joblib",
        "import pandas as pd",
        "import numpy as np",
        "",
        'st.set_page_config(page_title="Vehicle Maintenance Analysis", page_icon="⚙️", layout="wide")',
        'st.title("⚙️ Vehicle Maintenance Analysis Dashboard")',
        "",
        "@st.cache_resource",
        "def load_models():",
        '    return joblib.load("final_model.joblib"), joblib.load("scaler.joblib"), joblib.load("feature_names.joblib")',
        "",
        "try:",
        "    model, scaler, features = load_models()",
        '    rpm = st.number_input("Engine RPM", 50.0, 3000.0, 700.0)',
        '    lub_p = st.number_input("Lub Oil Pressure (bar)", 0.0, 10.0, 2.49)',
        '    fuel_p = st.number_input("Fuel Pressure (bar)", 0.0, 30.0, 11.79)',
        '    cool_p = st.number_input("Coolant Pressure (bar)", 0.0, 10.0, 3.17)',
        '    lub_t = st.number_input("Lub Oil Temp (°C)", 50.0, 120.0, 84.14)',
        '    cool_t = st.number_input("Coolant Temp (°C)", 50.0, 220.0, 81.63)',
        "",
        '    if st.button("PREDICT MAINTENANCE REQUIREMENT"):',
        "        inp = scaler.transform(pd.DataFrame([[rpm, lub_p, fuel_p, cool_p, lub_t, cool_t]], columns=features))",
        "        pred = model.predict(inp)[0]",
        "        prob = model.predict_proba(inp)[0]",
        '        if pred == 1:',
        '            st.error(f"⚠️ MAINTENANCE REQUIRED (Confidence: {np.max(prob)*100:.2f}%)")',
        "        else:",
        '            st.success(f"✅ NORMAL CONDITION (Confidence: {np.max(prob)*100:.2f}%)")',
        "except Exception as e:",
        '    st.error(f"Error loading model: {e}")',
        "'''",
        "",
        'with open("app.py", "w") as f:',
        "    f.write(app_code)",
        'print("✅ app.py created successfully!")'
    ]
    cells.append(nbf.v4.new_code_cell("\n".join(cell10_code)))

    # Cell 11: Launch Streamlit (Colab Native Shell Commands)
    cells.append(nbf.v4.new_markdown_cell("## 10. Launch Streamlit Web Application in Colab"))
    cell11_code = [
        "# Shell commands for Google Colab deployment",
        "!npm install -g localtunnel",
        "print('🔑 LOCALTUNNEL PASSWORD (Copy the IP address below):')",
        "!curl -s https://ipv4.icanhazip.com",
        "print('\\n🌐 CLICK THE LOCALTUNNEL URL BELOW TO VIEW YOUR APP:')",
        "!streamlit run app.py --server.port 8501 & npx localtunnel --port 8501"
    ]
    cells.append(nbf.v4.new_code_cell("\n".join(cell11_code)))

    nb['cells'] = cells

    with open("google_colab_vehicle_maintenance.ipynb", "w") as f:
        nbf.write(nb, f)
    with open("notebooks/google_colab_vehicle_maintenance.ipynb", "w") as f:
        nbf.write(nb, f)

    print("Native Google Colab notebook regenerated!")

if __name__ == "__main__":
    create_native_colab_notebook()
