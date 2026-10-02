# UNIVERSITY ACADEMIC PROJECT REPORT
## Case Study No. 67: Vehicle Maintenance Analysis and Prediction Using Machine Learning

---

### ABSTRACT
Modern commercial fleet management requires automated predictive maintenance systems to minimize unscheduled vehicle downtime, lower repair overheads, and improve highway safety. This project presents an end-to-end machine learning workflow applied to an engine health telemetry dataset comprising 19,535 records. Six operational sensor parameters—Engine RPM, Lubrication Oil Pressure, Fuel Pressure, Coolant Pressure, Lubrication Oil Temperature, and Coolant Temperature—were analyzed to classify engine operational state as either Normal (0) or Maintenance Required (1). Multiple supervised classification algorithms were trained and evaluated using 80/20 stratified train-test splitting and 5-fold cross-validation. Support Vector Machine (SVM) achieved the best performance with an F1-Score of 0.7709, Recall of 89.57%, and Accuracy of 66.44%. The model was serialized and integrated into an interactive web dashboard built with Streamlit.

---

### 1. INTRODUCTION
Predictive maintenance leverages sensor telemetry and artificial intelligence to evaluate machine health and forecast maintenance requirements before equipment breakdown occurs. In automotive engineering and commercial logistics, monitoring engine health metrics in real time provides actionable insights for maintenance dispatchers.

---

### 2. PROBLEM STATEMENT
"A fleet operator wants to study patterns that may indicate maintenance requirements."
The primary challenge is detecting complex multivariate sensor relationships that indicate impending engine degradation from operational telemetry.

---

### 3. OBJECTIVES
1. Conduct quantitative data verification and exploratory analysis on 19,535 engine sensor records.
2. Develop a reproducible, leakage-free preprocessing pipeline.
3. Train 5 baseline classification models: Logistic Regression, KNN, Decision Tree, Random Forest, and SVM.
4. Apply 5-fold Stratified Cross-Validation and hyperparameter optimization.
5. Evaluate models across Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices.
6. Deploy the selected model as a Streamlit web application.

---

### 4. DATASET & DATA DICTIONARY
- Total Records: 19,535
- Target Variable: `Engine Condition` (0 = Normal, 1 = Maintenance Required)
- Class Distribution: Normal = 7,218 (36.95%), Maintenance = 12,317 (63.05%)

| Feature | Data Type | Role | Summary Range |
| :--- | :--- | :--- | :--- |
| **Engine RPM** | `int64` | Input | 61 – 2,239 RPM |
| **Lub Oil Pressure** | `float64` | Input | 0.003 – 7.27 bar |
| **Fuel Pressure** | `float64` | Input | 0.003 – 21.14 bar |
| **Coolant Pressure** | `float64` | Input | 0.002 – 7.48 bar |
| **Lub Oil Temp** | `float64` | Input | 71.32 – 89.58 °C |
| **Coolant Temp** | `float64` | Input | 61.67 – 195.53 °C |
| **Engine Condition** | `int64` | Target | Binary {0, 1} |

---

### 5. PREPROCESSING & METHODOLOGY
1. **Cleaning**: Verification confirmed 0 missing values and 0 duplicate rows.
2. **Train-Test Split**: Stratified 80/20 split (`X_train`: 15,628, `X_test`: 3,907).
3. **Scaling**: `StandardScaler` fitted exclusively on `X_train` to eliminate data leakage.

---

### 6. EXPERIMENTAL RESULTS & MODEL COMPARISON

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine** | **0.6644** | **0.6767** | **0.8957** | **0.7709** | **0.6766** |
| **Logistic Regression** | 0.6611 | 0.6787 | 0.8782 | 0.7657 | 0.6918 |
| **Random Forest** | 0.6619 | 0.6901 | 0.8417 | 0.7584 | 0.6957 |
| **Tuned Random Forest** | 0.6583 | 0.6874 | 0.8400 | 0.7561 | 0.6988 |
| **Decision Tree** | 0.6376 | 0.6976 | 0.7503 | 0.7230 | 0.6397 |
| **K-Nearest Neighbors** | 0.6261 | 0.6822 | 0.7617 | 0.7197 | 0.6175 |

---

### 7. ERROR ANALYSIS
For the final SVM model on the test dataset (3,907 samples):
- True Positives (TP): 2,206
- True Negatives (TN): 390
- False Positives (FP): 1,054
- False Negatives (FN): 257

The model achieves a high Recall of 89.57%, ensuring that nearly 9 out of 10 engines in need of maintenance are correctly flagged, keeping uncaught engine breakdowns (false negatives) low (10.43%).

---

### 8. CONCLUSION & FUTURE SCOPE
The predictive maintenance solution successfully classifies engine health using sensor telemetry. Support Vector Machine emerged as the optimal model and was deployed via Streamlit. Future work includes time-series Remaining Useful Life (RUL) estimation using recurrent neural networks and real-time MQTT IoT telemetry streaming.
