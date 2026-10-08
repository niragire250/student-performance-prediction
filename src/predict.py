"""
predict.py
-----------
Loads the saved model pipeline and produces a prediction for a single new
student record. This module is imported by app.py (the Streamlit app) so
that the exact same feature-engineering logic used in training is reused
at prediction time -- eliminating any train/serve mismatch.
"""

import json

import joblib
import pandas as pd

from config import (CATEGORY_VALUES_PATH, METADATA_PATH, MODEL_PATH,
                     REQUIRED_RAW_FIELDS, get_logger)
from feature_engineering import engineer_features

logger = get_logger(__name__)

_model = None
_metadata = None
_category_values = None


def _lazy_load():
    global _model, _metadata, _category_values
    if _model is None:
        logger.info(f"Loading model from {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
        with open(METADATA_PATH) as f:
            _metadata = json.load(f)
        with open(CATEGORY_VALUES_PATH) as f:
            _category_values = json.load(f)
    return _model, _metadata, _category_values


def get_metadata():
    _, meta, _ = _lazy_load()
    return meta


def get_category_values():
    _, _, cats = _lazy_load()
    return cats


def predict_single(raw_input: dict) -> dict:
    """
    raw_input must contain the RAW (pre-engineering) fields the UI collects,
    e.g. {"age": 17, "studytime": 2, "failures": 0, "G1": 12, "G2": 13, ...}

    Returns a dict with the predicted label, class probabilities, risk level,
    and a short plain-language interpretation.
    """
    model, meta, _ = _lazy_load()

    # Validate BEFORE feature engineering, so a missing raw field produces a
    # clean, user-facing ValueError instead of an internal KeyError.
    missing_raw = [c for c in REQUIRED_RAW_FIELDS if c not in raw_input
                   or raw_input[c] is None or raw_input[c] == ""]
    if missing_raw:
        raise ValueError(f"Missing required input fields: {missing_raw}")

    df = pd.DataFrame([raw_input])
    df = engineer_features(df)

    feature_cols = meta["numeric_features"] + meta["categorical_features"]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required input fields: {missing}")

    X = df[feature_cols]
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confidence = float(proba[pred])

    label = "PASS" if pred == 1 else "FAIL"
    prob_pass = float(proba[1])
    risk_level = calculate_risk_level(prob_pass, label)
    
    return {
        "prediction": label,
        "confidence": round(confidence * 100, 1),
        "probability_pass": round(prob_pass * 100, 1),
        "probability_fail": round(float(proba[0]) * 100, 1),
        "risk_level": risk_level,
    }


def calculate_risk_level(prob_pass: float, prediction: str) -> str:
    """
    Calculate risk level based on probability of passing.
    - Low risk: prob_pass >= 70%
    - Medium risk: 40% <= prob_pass < 70%
    - High risk: prob_pass < 40%
    """
    if prediction == "PASS":
        if prob_pass >= 0.70:
            return "Low"
        elif prob_pass >= 0.40:
            return "Medium"
        else:
            return "High"
    else:
        if prob_pass < 0.30:
            return "High"
        elif prob_pass < 0.60:
            return "Medium"
        else:
            return "Low"


def predict_batch(raw_inputs: list) -> list:
    """
    Predict for multiple students at once.
    
    Args:
        raw_inputs: List of dicts, each containing the RAW (pre-engineering) fields
        
    Returns:
        List of prediction dicts with prediction, confidence, probabilities, and risk level
    """
    model, meta, _ = _lazy_load()
    
    results = []
    for raw_input in raw_inputs:
        try:
            # Validate each input
            missing_raw = [c for c in REQUIRED_RAW_FIELDS if c not in raw_input
                           or raw_input[c] is None or raw_input[c] == ""]
            if missing_raw:
                results.append({
                    "error": f"Missing required fields: {missing_raw}",
                    "prediction": "ERROR"
                })
                continue
            
            df = pd.DataFrame([raw_input])
            df = engineer_features(df)
            
            feature_cols = meta["numeric_features"] + meta["categorical_features"]
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                results.append({
                    "error": f"Missing required fields: {missing}",
                    "prediction": "ERROR"
                })
                continue
            
            X = df[feature_cols]
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            confidence = float(proba[pred])
            
            label = "PASS" if pred == 1 else "FAIL"
            prob_pass = float(proba[1])
            risk_level = calculate_risk_level(prob_pass, label)
            
            results.append({
                "prediction": label,
                "confidence": round(confidence * 100, 1),
                "probability_pass": round(prob_pass * 100, 1),
                "probability_fail": round(float(proba[0]) * 100, 1),
                "risk_level": risk_level,
            })
        except Exception as e:
            results.append({
                "error": str(e),
                "prediction": "ERROR"
            })
    
    return results


def get_feature_importance() -> dict:
    """
    Return the top feature importances from the model metadata.
    """
    _, meta, _ = _lazy_load()
    return meta.get("top_feature_importances", {})


def get_intervention_recommendations(student_data: dict, prediction_result: dict) -> list:
    """
    Generate practical intervention recommendations based on student data
    and prediction result.
    
    Args:
        student_data: Raw student input dict
        prediction_result: Result from predict_single
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # Risk-based recommendations
    if prediction_result.get("risk_level") == "High":
        recommendations.append("Schedule immediate one-on-one counseling session with student")
        recommendations.append("Review academic support options and tutoring availability")
    
    # Attendance-based
    if student_data.get("absences", 0) > 10:
        recommendations.append("Address attendance patterns - discuss barriers to attendance")
        recommendations.append("Consider attendance improvement plan")
    
    # Study time
    if student_data.get("studytime", 2) < 2:
        recommendations.append("Encourage increased study time - recommend study skills workshop")
    
    # Past failures
    if student_data.get("failures", 0) > 0:
        recommendations.append("Review past failure patterns and identify root causes")
        recommendations.append("Consider remedial support in challenging subjects")
    
    # Grade trend
    g1 = student_data.get("G1", 0)
    g2 = student_data.get("G2", 0)
    if g2 < g1:
        recommendations.append("Grade declining - investigate recent challenges or changes")
    
    # Support systems
    if student_data.get("schoolsup") == "no":
        recommendations.append("Consider enrolling student in extra school support program")
    
    if student_data.get("famsup") == "no":
        recommendations.append("Discuss family educational support options with parents/guardians")
    
    # Family relationship
    if student_data.get("famrel", 5) < 3:
        recommendations.append("Family relationship quality low - consider family counseling referral")
    
    # Alcohol consumption
    if student_data.get("Walc", 1) >= 4 or student_data.get("Dalc", 1) >= 3:
        recommendations.append("High alcohol consumption detected - consider wellness intervention")
    
    # Higher education goals
    if student_data.get("higher") == "no":
        recommendations.append("Discuss future educational and career aspirations with student")
    
    if not recommendations:
        recommendations.append("Student appears on track - continue monitoring and encouragement")
    
    return recommendations


if __name__ == "__main__":
    # Quick manual smoke test using a plausible student record
    sample = {
        "school": "GP", "sex": "F", "age": 16, "address": "U", "famsize": "GT3",
        "Pstatus": "T", "Medu": 3, "Fedu": 2, "Mjob": "services", "Fjob": "other",
        "reason": "course", "guardian": "mother", "traveltime": 1, "studytime": 2,
        "failures": 0, "schoolsup": "no", "famsup": "yes", "paid": "no",
        "activities": "yes", "nursery": "yes", "higher": "yes", "internet": "yes",
        "romantic": "no", "famrel": 4, "freetime": 3, "goout": 3, "Dalc": 1,
        "Walc": 1, "health": 4, "absences": 4, "G1": 12, "G2": 13,
    }
    result = predict_single(sample)
    logger.info(f"Smoke-test prediction: {result}")
