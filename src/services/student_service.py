"""
student_service.py
------------------
Business logic for student operations.
Bridges the UI layer with the data repositories.
"""

from typing import Optional, Dict, Any, List
from data.repositories import StudentRepository, InstitutionRepository
from config import REQUIRED_RAW_FIELDS, get_logger

logger = get_logger(__name__)


class StudentService:
    """Service for student-related business logic."""
    
    @staticmethod
    def get_institutions() -> List[Dict[str, Any]]:
        """Get list of active institutions for dropdown."""
        return InstitutionRepository.get_all_active()
    
    @staticmethod
    def find_student_by_registration(registration_number: str, 
                                    institution_id: str) -> Dict[str, Any]:
        """
        Find student by registration number.
        
        Args:
            registration_number: Student's registration number
            institution_id: Institution ID
            
        Returns:
            Dict with student data and error/success status
            
        Raises:
            ValueError: If registration number is empty or invalid
        """
        # Validate input
        if not registration_number or not registration_number.strip():
            raise ValueError("Please enter a Registration Number.")
        
        registration_number = registration_number.strip()
        
        if len(registration_number) < 3:
            raise ValueError("Registration Number is too short.")
        
        # Check institution exists
        institution = InstitutionRepository.get_by_id(institution_id)
        if not institution:
            raise ValueError(f"Institution '{institution_id}' not found.")
        
        # Look up student
        student = StudentRepository.get_full_student_record(
            registration_number, institution_id
        )
        
        if not student:
            raise ValueError(
                f"No student found for Registration Number '{registration_number}' "
                f"at institution '{institution['institution_name']}'."
            )
        
        return student
    
    @staticmethod
    def map_to_ml_features(student_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map database student record to ML model features.
        
        Args:
            student_data: Student record from database
            
        Returns:
            Dict with REQUIRED_RAW_FIELDS for prediction
            
        Raises:
            ValueError: If required fields are missing
        """
        # Extract academic features from student data
        # The database fields should match the ML model's expected field names
        ml_features = {}
        
        # Check which required fields are present
        missing_fields = []
        for field in REQUIRED_RAW_FIELDS:
            if field in student_data and student_data[field] is not None:
                ml_features[field] = student_data[field]
            else:
                missing_fields.append(field)
        
        if missing_fields:
            raise ValueError(
                f"Student record is missing required academic information: {missing_fields}. "
                "Prediction cannot be performed without complete data."
            )
        
        return ml_features
    
    @staticmethod
    def validate_for_prediction(student_data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate that student data has all required fields for prediction.
        
        Args:
            student_data: Student record
            
        Returns:
            Tuple of (is_valid, list_of_missing_fields)
        """
        missing = [f for f in REQUIRED_RAW_FIELDS 
                  if f not in student_data or student_data[f] is None]
        return (len(missing) == 0, missing)
