"""
Prediction Module for Vehicle Maintenance Analysis.
Loads the trained model and scaler to make predictions on engine parameter inputs.
Includes safety bounds validation for user input parameters.
"""

import os
import joblib
import pandas as pd
import numpy as np

# Typical safe range definitions for input validation
PARAM_BOUNDS = {
    "Engine RPM": (50.0, 3000.0),
    "Lub Oil Pressure": (0.0, 10.0),
    "Fuel Pressure": (0.0, 30.0),
    "Coolant Pressure": (0.0, 10.0),
    "Lub Oil Temp": (50.0, 120.0),
    "Coolant Temp": (50.0, 220.0)
}

def load_prediction_artifacts(models_dir="models"):
    """Load trained model, scaler, and feature names."""
    model_path = os.path.join(models_dir, "final_model.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    features_path = os.path.join(models_dir, "feature_names.joblib")
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path)):
        raise FileNotFoundError("Trained model or scaler artifacts missing in models/ directory. Run train_models.py first.")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path)
    return model, scaler, feature_names

def validate_input_params(input_dict):
    """
    Validate numerical input parameter dictionary.
    Returns (is_valid, list_of_warning_messages).
    """
    warnings = []
    for param, val in input_dict.items():
        if param in PARAM_BOUNDS:
            min_val, max_val = PARAM_BOUNDS[param]
            if val < min_val or val > max_val:
                warnings.append(f"Warning: '{param}' value {val} is outside expected range ({min_val} - {max_val}).")
    return len(warnings) == 0, warnings

def predict_engine_condition(input_dict, models_dir="models"):
    """
    Predict engine condition based on user input parameters.
    Returns prediction label, class probability, and validation warnings.
    """
    model, scaler, feature_names = load_prediction_artifacts(models_dir)
    
    # Construct DataFrame with matching feature order
    input_df = pd.DataFrame([input_dict])[feature_names]
    
    # Check input bounds
    is_valid, warnings = validate_input_params(input_dict)
    
    # Scale inputs using fitted scaler
    input_scaled = scaler.transform(input_df)
    
    # Predict label & probability
    prediction = int(model.predict(input_scaled)[0])
    probabilities = model.predict_proba(input_scaled)[0]
    
    label_map = {
        0: "Normal Condition",
        1: "Maintenance Required"
    }
    
    return {
        "prediction_code": prediction,
        "prediction_label": label_map[prediction],
        "normal_probability": float(probabilities[0]),
        "maintenance_probability": float(probabilities[1]),
        "confidence_score": float(np.max(probabilities) * 100),
        "is_within_normal_bounds": is_valid,
        "warnings": warnings
    }

if __name__ == "__main__":
    sample_input = {
        "Engine RPM": 700.0,
        "Lub Oil Pressure": 2.49,
        "Fuel Pressure": 11.79,
        "Coolant Pressure": 3.17,
        "Lub Oil Temp": 84.14,
        "Coolant Temp": 81.63
    }
    res = predict_engine_condition(sample_input)
    print("Sample Prediction Result:", res)
