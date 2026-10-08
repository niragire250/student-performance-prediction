"""
schema.py
---------
Central feature schema definition.
Single source of truth for model feature requirements.
"""

from typing import Dict, List, Set, Tuple
from config import REQUIRED_RAW_FIELDS, get_logger

logger = get_logger(__name__)

# Feature types
FEATURE_TYPES = {
    # Numeric features
    "age": "numeric",
    "Medu": "numeric",
    "Fedu": "numeric",
    "traveltime": "numeric",
    "studytime": "numeric",
    "failures": "numeric",
    "famrel": "numeric",
    "freetime": "numeric",
    "goout": "numeric",
    "Dalc": "numeric",
    "Walc": "numeric",
    "health": "numeric",
    "absences": "numeric",
    "G1": "numeric",
    "G2": "numeric",
    "G3": "numeric",
    
    # Categorical features
    "school": "categorical",
    "sex": "categorical",
    "address": "categorical",
    "famsize": "categorical",
    "Pstatus": "categorical",
    "Mjob": "categorical",
    "Fjob": "categorical",
    "reason": "categorical",
    "guardian": "categorical",
    "schoolsup": "categorical",
    "famsup": "categorical",
    "paid": "categorical",
    "activities": "categorical",
    "nursery": "categorical",
    "higher": "categorical",
    "internet": "categorical",
    "romantic": "categorical",
}

# Numeric features list
NUMERIC_FEATURES = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2", "G3"
]

# Categorical features list
CATEGORICAL_FEATURES = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities", "nursery",
    "higher", "internet", "romantic"
]


def validate_features(features: Dict) -> Tuple[bool, List[str]]:
    """
    Validate that features match the required schema.
    
    Args:
        features: Dictionary of feature names to values
        
    Returns:
        Tuple of (is_valid, list_of_missing_fields)
    """
    missing = []
    for field in REQUIRED_RAW_FIELDS:
        if field not in features or features[field] is None:
            missing.append(field)
    
    if missing:
        logger.warning(f"Missing required fields: {missing}")
    
    return (len(missing) == 0, missing)


def get_feature_type(field_name: str) -> str:
    """
    Get the type of a feature.
    
    Args:
        field_name: Name of the feature
        
    Returns:
        "numeric", "categorical", or "unknown"
    """
    return FEATURE_TYPES.get(field_name, "unknown")
