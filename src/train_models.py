"""
Model Training and Evaluation Module for Vehicle Maintenance Analysis.
Trains baseline and tuned models, calculates actual evaluation metrics,
performs cross-validation, hyperparameter tuning, saves plots and model artifacts.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import joblib

def train_and_evaluate_all(X_train_scaled, X_test_scaled, y_train, y_test, feature_names):
    """
    Train 5 classification models, run Stratified 5-fold CV, evaluate on test set,
    tune hyper-parameters, select best model, save metrics & artifacts.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)
    
    # Define baseline classifiers
    classifiers = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=12),
        "Support Vector Machine": SVC(random_state=42, probability=True, kernel='rbf')
    }
    
    results = {}
    fitted_models = {}
    cv_results_summary = {}
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("--- 1. Training Baseline Models & Running 5-Fold Cross-Validation ---")
    for name, clf in classifiers.items():
        print(f"Training {name}...")
        
        # 5-Fold Cross Validation on Training Data
        cv_scores = cross_validate(clf, X_train_scaled, y_train, cv=skf, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])
        
        cv_results_summary[name] = {
            "cv_accuracy_mean": float(np.mean(cv_scores['test_accuracy'])),
            "cv_accuracy_std": float(np.std(cv_scores['test_accuracy'])),
            "cv_f1_mean": float(np.mean(cv_scores['test_f1'])),
            "cv_roc_auc_mean": float(np.mean(cv_scores['test_roc_auc']))
        }
        
        # Fit on full training set
        clf.fit(X_train_scaled, y_train)
        fitted_models[name] = clf
        
        # Predict on Test Set
        y_pred = clf.predict(X_test_scaled)
        y_proba = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, "predict_proba") else None
        
        # Compute exact metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        results[name] = {
            "Accuracy": float(acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1-Score": float(f1),
            "ROC-AUC": float(auc),
            "Confusion Matrix": {
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp)
            },
            "CV Scores": cv_results_summary[name]
        }

    # --- 2. Hyperparameter Tuning for Top Model (Random Forest) ---
    print("\n--- 2. Hyperparameter Tuning for Random Forest ---")
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=skf,
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    best_rf = grid_search.best_estimator_
    
    # Evaluate Tuned Random Forest
    y_pred_tuned = best_rf.predict(X_test_scaled)
    y_proba_tuned = best_rf.predict_proba(X_test_scaled)[:, 1]
    
    acc_t = accuracy_score(y_test, y_pred_tuned)
    prec_t = precision_score(y_test, y_pred_tuned)
    rec_t = recall_score(y_test, y_pred_tuned)
    f1_t = f1_score(y_test, y_pred_tuned)
    auc_t = roc_auc_score(y_test, y_proba_tuned)
    cm_t = confusion_matrix(y_test, y_pred_tuned)
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
    
    results["Tuned Random Forest"] = {
        "Accuracy": float(acc_t),
        "Precision": float(prec_t),
        "Recall": float(rec_t),
        "F1-Score": float(f1_t),
        "ROC-AUC": float(auc_t),
        "Confusion Matrix": {
            "TN": int(tn_t),
            "FP": int(fp_t),
            "FN": int(fn_t),
            "TP": int(tp_t)
        },
        "Best Params": grid_search.best_params_
    }
    fitted_models["Tuned Random Forest"] = best_rf
    
    # --- 3. Model Selection ---
    # Select best model based on F1-Score / Accuracy on Test Set
    best_model_name = max(results.keys(), key=lambda k: results[k]["F1-Score"])
    best_model = fitted_models[best_model_name]
    print(f"\nSelected Best Model: {best_model_name} (F1-Score: {results[best_model_name]['F1-Score']:.4f})")
    
    # Save best model to joblib
    joblib.dump(best_model, "models/final_model.joblib")
    
    # Save overall metrics to JSON
    with open("outputs/metrics/model_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # --- 4. Plot Model Comparison ---
    comp_df = pd.DataFrame([
        {
            "Model": m,
            "Accuracy": results[m]["Accuracy"],
            "Precision": results[m]["Precision"],
            "Recall": results[m]["Recall"],
            "F1-Score": results[m]["F1-Score"],
            "ROC-AUC": results[m]["ROC-AUC"]
        }
        for m in results.keys()
    ])
    comp_df.to_csv("outputs/metrics/model_comparison.csv", index=False)
    
    # Save Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    melted = comp_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    sns.barplot(data=melted, x="Model", y="Score", hue="Metric", ax=ax, palette="mako")
    ax.set_ylim(0.7, 1.02)
    ax.set_title("Model Comparison Across Key Classification Metrics", fontweight="bold", pad=15)
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig("outputs/figures/model_comparison.png", dpi=300)
    plt.close(fig)
    
    # Save Confusion Matrices Chart
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for idx, (m_name, m_res) in enumerate(results.items()):
        if idx < len(axes):
            cm_matrix = np.array([
                [m_res["Confusion Matrix"]["TN"], m_res["Confusion Matrix"]["FP"]],
                [m_res["Confusion Matrix"]["FN"], m_res["Confusion Matrix"]["TP"]]
            ])
            sns.heatmap(cm_matrix, annot=True, fmt="d", cmap="Blues", ax=axes[idx], cbar=False,
                        xticklabels=["Normal", "Maintenance"], yticklabels=["Normal", "Maintenance"])
            axes[idx].set_title(f"{m_name}", fontweight="bold")
            axes[idx].set_ylabel("Actual Label")
            axes[idx].set_xlabel("Predicted Label")
    plt.tight_layout()
    fig.savefig("outputs/figures/confusion_matrices.png", dpi=300)
    plt.close(fig)
    
    # ROC Curves Plot
    fig, ax = plt.subplots(figsize=(9, 7))
    for m_name, model in fitted_models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            ax.plot(fpr, tpr, label=f"{m_name} (AUC = {results[m_name]['ROC-AUC']:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig("outputs/figures/roc_curves.png", dpi=300)
    plt.close(fig)
    
    # Feature Importance (if tree-based)
    if hasattr(best_model, "feature_importances_"):
        fig, ax = plt.subplots(figsize=(8, 5))
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], ax=ax, palette="viridis")
        ax.set_title(f"Feature Importance ({best_model_name})", fontweight="bold")
        ax.set_xlabel("Relative Importance")
        plt.tight_layout()
        fig.savefig("outputs/figures/feature_importance.png", dpi=300)
        plt.close(fig)

    print("Model training, cross-validation, tuning, evaluation & plotting completed successfully!")
    return results, best_model_name

if __name__ == "__main__":
    from data_preprocessing import load_data, clean_data, prepare_pipeline
    df = load_data()
    cleaned_df, _ = clean_data(df)
    pipe = prepare_pipeline(cleaned_df)
    results, best_model_name = train_and_evaluate_all(
        pipe["X_train_scaled"], pipe["X_test_scaled"],
        pipe["y_train"], pipe["y_test"], pipe["feature_names"]
    )
