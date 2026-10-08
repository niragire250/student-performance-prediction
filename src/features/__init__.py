"""
features package
----------------
Feature schema and mapping layer for RP Student Success System.
Separates RP student records from ML model feature requirements.
"""

from .schema import (
    REQUIRED_RAW_FIELDS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_TYPES,
    validate_features
)
from .mapper import (
    RPFeatureMapper,
    map_rp_student_to_legacy_features,
    map_rp_student_to_rp_features
)

__all__ = [
    'REQUIRED_RAW_FIELDS',
    'NUMERIC_FEATURES',
    'CATEGORICAL_FEATURES',
    'FEATURE_TYPES',
    'validate_features',
    'RPFeatureMapper',
    'map_rp_student_to_legacy_features',
    'map_rp_student_to_rp_features'
]
