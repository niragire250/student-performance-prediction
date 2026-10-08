"""
mapper.py
---------
Feature mapping layer.
Maps RP student records to ML model feature requirements.
Separates RP institution identity from model compatibility.
"""

from typing import Dict, Any, Optional
from .schema import REQUIRED_RAW_FIELDS, validate_features
from config import LEGACY_MODEL_COMPATIBILITY, get_logger

logger = get_logger(__name__)


class RPFeatureMapper:
    """
    Maps RP student records to ML model features.
    
    This layer separates:
    - RP institution identity (RP-GISHARI, RP-KIGALI, etc.)
    - ML model compatibility (GP, MS for legacy model)
    
    In production with an RP-trained model, this would map to RP-specific
    features instead of legacy Portuguese school features.
    """
    
    @staticmethod
    def map_rp_student_to_legacy_features(rp_student: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map RP student record to legacy model features.
        
        The legacy model was trained on Portuguese secondary school data.
        This mapping provides compatibility for prototype demonstration.
        
        Args:
            rp_student: RP student record from database
            
        Returns:
            Dictionary with REQUIRED_RAW_FIELDS for legacy model
            
        Raises:
            ValueError: If required fields are missing
        """
        # Extract academic features from RP student record
        # The RP database stores legacy model features in student_academic_records
        ml_features = {}
        
        # Check which required fields are present
        missing_fields = []
        for field in REQUIRED_RAW_FIELDS:
            if field in rp_student and rp_student[field] is not None:
                ml_features[field] = rp_student[field]
            else:
                missing_fields.append(field)
        
        if missing_fields:
            raise ValueError(
                f"RP student record is missing required academic information: {missing_fields}. "
                "Prediction cannot be performed without complete data."
            )
        
        logger.info(f"Mapped RP student to legacy features successfully")
        return ml_features
    
    @staticmethod
    def map_institution_to_legacy_school(institution_id: str) -> str:
        """
        Map RP institution ID to legacy model school code.
        
        Args:
            institution_id: RP institution ID (e.g., RP-KIGALI)
            
        Returns:
            Legacy school code (GP or MS)
        """
        return LEGACY_MODEL_COMPATIBILITY.get(institution_id, "GP")
    
    @staticmethod
    def map_rp_student_to_rp_features(rp_student: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map RP student record to RP-specific model features.
        
        This is a placeholder for when an RP-trained model is available.
        Currently returns the legacy features for compatibility.
        
        Args:
            rp_student: RP student record from database
            
        Returns:
            Dictionary with RP-specific features (placeholder)
        """
        # TODO: Implement when RP-trained model is available
        # For now, use legacy features
        logger.warning("RP-trained model not yet available, using legacy features")
        return RPFeatureMapper.map_rp_student_to_legacy_features(rp_student)


def map_rp_student_to_legacy_features(rp_student: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function for mapping RP student to legacy features.
    """
    return RPFeatureMapper.map_rp_student_to_legacy_features(rp_student)


def map_rp_student_to_rp_features(rp_student: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function for mapping RP student to RP features.
    """
    return RPFeatureMapper.map_rp_student_to_rp_features(rp_student)
