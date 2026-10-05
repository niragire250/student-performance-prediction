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

    Returns a dict with the predicted label, class probabilities and a short
    plain-language interpretation.
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
    return {
        "prediction": label,
        "confidence": round(confidence * 100, 1),
        "probability_pass": round(float(proba[1]) * 100, 1),
        "probability_fail": round(float(proba[0]) * 100, 1),
    }


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
