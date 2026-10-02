# Viva Preparation Guide — Vehicle Maintenance Analysis

> **Quick Study Guide for Academic Viva & Oral Examination**

---

## 1. DATASET QUESTIONS

### Q1: What dataset did you use for this project?
**Answer**: I used the *Automotive Vehicles Engine Health Dataset*, which contains sensor telemetry collected from vehicle engines to determine predictive maintenance requirements.

### Q2: How many records and features are in the dataset?
**Answer**: The dataset contains **19,535 records** and **7 total columns**: 6 numerical input features and 1 binary target variable (`Engine Condition`).

### Q3: What are the input features?
**Answer**:
1. **Engine RPM**: Rotational speed of the engine.
2. **Lub Oil Pressure**: Lubrication oil pressure (bar).
3. **Fuel Pressure**: Fuel delivery pressure (bar).
4. **Coolant Pressure**: Cooling system pressure (bar).
5. **Lub Oil Temp**: Lubrication oil temperature (°C).
6. **Coolant Temp**: Engine coolant temperature (°C).

### Q4: What is the target variable and what does it represent?
**Answer**: The target variable is `Engine Condition`. It is a binary classification label:
- `0` = Normal Operating Condition (36.95% of dataset).
- `1` = Maintenance Required (63.05% of dataset).

---

## 2. PREPROCESSING & DATA CLEANING QUESTIONS

### Q5: Why did you perform data cleaning?
**Answer**: To check for missing values, duplicate records, and invalid data types. In our dataset, there were 0 missing values and 0 duplicate rows.

### Q6: What is feature scaling and why is it necessary?
**Answer**: Feature scaling standardizes numerical feature ranges (mean=0, variance=1) using `StandardScaler`. It is necessary because features like Engine RPM (ranging up to 2,239) operate on much larger scales than Lub Oil Pressure (ranging from 0 to 7), which would otherwise dominate distance-based algorithms like KNN and SVM.

### Q7: How did you perform the Train-Test Split and why?
**Answer**: I performed an **80/20 stratified train-test split** (`random_state=42`). 80% (15,628 records) was used for model training, and 20% (3,907 records) was reserved for testing. Stratification maintains the exact class proportion (63% / 37%) in both sets.

### Q8: What is data leakage and how did you prevent it?
**Answer**: Data leakage occurs when information from the test dataset influences model training. I prevented it by fitting the `StandardScaler` **exclusively on the training dataset** (`X_train`), and then transforming both `X_train` and `X_test` using that fitted scaler.

---

## 3. MACHINE LEARNING MODEL QUESTIONS

### Q9: Why is this a classification problem instead of regression?
**Answer**: Because the target variable `Engine Condition` consists of discrete categorical classes (`0 = Normal`, `1 = Maintenance Required`), rather than a continuous numerical quantity.

### Q10: Which algorithms did you evaluate?
**Answer**:
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
5. Support Vector Machine (SVM)

### Q11: Briefly explain how Support Vector Machine (SVM) works.
**Answer**: SVM finds an optimal hyperplane that maximizes the margin of separation between the two classes in a high-dimensional feature space using kernel functions (such as RBF).

### Q12: What is overfitting and underfitting?
**Answer**:
- **Overfitting**: When a model learns noise and training data details too closely, resulting in high training accuracy but poor generalization on test data.
- **Underfitting**: When a model is too simple to capture underlying patterns, performing poorly on both training and testing sets.

### Q13: What is Cross-Validation?
**Answer**: Cross-validation (specifically Stratified 5-Fold CV) splits the training data into 5 equal folds. The model is trained on 4 folds and tested on the remaining fold 5 times, providing an unbiased estimate of generalization stability.

---

## 4. EVALUATION & RESULTS QUESTIONS

### Q14: What metrics did you use to evaluate models?
**Answer**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.

### Q15: Explain Precision vs Recall in the context of vehicle maintenance.
**Answer**:
- **Precision**: Of all vehicles predicted as needing maintenance, what percentage actually required maintenance? (Avoids false alarms / unnecessary repair costs).
- **Recall**: Of all vehicles that actually needed maintenance, what percentage did the model correctly identify? (Avoids uncaught engine breakdowns).
- **F1-Score**: The harmonic mean of Precision and Recall.

### Q16: What does a False Negative mean in vehicle maintenance?
**Answer**: A **False Negative** means the model predicts an engine is *Normal* when it actually *requires maintenance*. In real-world fleet management, false negatives are dangerous as they can lead to catastrophic engine breakdown on the road.

### Q17: Which model performed best and why did you select it?
**Answer**: **Support Vector Machine (SVM)** performed best. It achieved the highest test **F1-Score (0.7709)**, highest **Recall (89.57%)**, and highest overall **Accuracy (66.44%)**, successfully identifying 2,206 out of 2,463 actual maintenance cases in the test dataset.

---

## 5. STREAMLIT & APPLICATION QUESTIONS

### Q18: How does the Streamlit web application work?
**Answer**: The Streamlit app provides an interactive user interface where fleet engineers enter 6 engine sensor inputs. The app loads the saved `final_model.joblib` and `scaler.joblib` pipelines, transforms the inputs, predicts the engine condition label, and displays probability breakdowns with confidence scores.
