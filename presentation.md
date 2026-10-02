# Presentation Slides: Vehicle Maintenance Analysis & Prediction
## Case Study No. 67 — Machine Learning Exam Project

---

### Slide 1: Title Slide
- **Project Title**: Vehicle Maintenance Analysis and Prediction Using Machine Learning
- **Subtitle**: Machine Learning Based Predictive Engine Maintenance System
- **Domain**: Predictive Maintenance / Fleet Telemetry Analytics

---

### Slide 2: Problem Statement & Motivation
- Commercial fleet operators face costly downtime due to sudden engine failures.
- Reactive maintenance causes revenue loss and highway safety risks.
- **Goal**: Analyze engine telemetry sensor data to predict maintenance requirements before breakdown occurs.

---

### Slide 3: Project Objectives
- Analyze 19,535 engine sensor telemetry records.
- Build a robust, leakage-free ML pipeline (cleaning, scaling, stratified splitting).
- Evaluate 5 algorithms: Logistic Regression, KNN, Decision Tree, Random Forest, SVM.
- Deploy an interactive Streamlit web dashboard.

---

### Slide 4: Dataset Overview
- **Records**: 19,535 rows
- **Input Features (6)**: Engine RPM, Lub Oil Pressure, Fuel Pressure, Coolant Pressure, Lub Oil Temp, Coolant Temp.
- **Target Variable**: Engine Condition (`0 = Normal`, `1 = Maintenance Required`).
- **Class Balance**: 63.05% Maintenance Required, 36.95% Normal.

---

### Slide 5: Exploratory Data Analysis (EDA) Highlights
- **Correlation**: Oil Pressure and Coolant Pressure exhibit negative correlation with high operating temperatures.
- **Outliers**: Identified operating extremes in RPM (up to 2,239 RPM) and Coolant Temp (up to 195.5 °C).

---

### Slide 6: Machine Learning Methodology
- **Train-Test Split**: 80% Train (15,628 rows) / 20% Test (3,907 rows) with stratification.
- **Preprocessing**: `StandardScaler` fitted on training set only (preventing data leakage).
- **Validation**: 5-Fold Stratified Cross-Validation & GridSearchCV hyperparameter tuning.

---

### Slide 7: Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SVM (Selected)** | **0.6644** | **0.6767** | **0.8957** | **0.7709** | **0.6766** |
| **Logistic Regression** | 0.6611 | 0.6787 | 0.8782 | 0.7657 | 0.6918 |
| **Random Forest** | 0.6619 | 0.6901 | 0.8417 | 0.7584 | 0.6957 |
| **Decision Tree** | 0.6376 | 0.6976 | 0.7503 | 0.7230 | 0.6397 |
| **KNN** | 0.6261 | 0.6822 | 0.7617 | 0.7197 | 0.6175 |

---

### Slide 8: Model Selection & Justification
- **Selected Model**: Support Vector Machine (SVM)
- **Rationale**: Highest test F1-Score (**0.7709**) and highest Recall (**89.57%**).
- **Practical Impact**: High Recall minimizes dangerous uncaught engine breakdowns.

---

### Slide 9: Error Analysis
- **True Positives (TP)**: 2,206 maintenance cases correctly flagged.
- **False Negatives (FN)**: 257 uncaught cases (10.43% false negative rate).
- **Trade-off**: High recall prioritizes vehicle safety over minor false alarm inspection costs.

---

### Slide 10: Streamlit Web Dashboard
- Interactive inputs for all 6 engine telemetry sensors.
- Real-time classification result (Normal vs. Maintenance Required).
- Confidence scores and probability progress bars.
- Live bounds validation warnings.

---

### Slide 11: Limitations & Future Scope
- **Limitations**: Bounded by sensor training ranges; binary target label.
- **Future Scope**: Time-series LSTM models for Remaining Useful Life (RUL) estimation & IoT MQTT integration.

---

### Slide 12: Conclusion & Q&A
- Developed complete, exam-ready predictive maintenance ML pipeline.
- Achieved robust classification using SVM (F1: 0.7709, Recall: 89.57%).
- Successfully deployed Streamlit application.
- Thank you — Open for Questions!
