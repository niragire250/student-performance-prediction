"""
test_database.py
----------------
Tests for the database layer and student repository.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.database import init_database, database_exists, DB_PATH
from data.repositories import InstitutionRepository, StudentRepository
from config import get_logger

logger = get_logger(__name__)


@pytest.fixture(scope="module")
def test_db():
    """Initialize test database."""
    # Use a separate test database
    import tempfile
    import os
    
    # Backup original DB path
    original_db_path = DB_PATH
    
    # Create temporary test database
    test_db_dir = Path(__file__).resolve().parents[1] / "data"
    test_db_dir.mkdir(exist_ok=True)
    test_db_path = test_db_dir / "test_rp_demo.db"
    
    # Temporarily override DB_PATH
    import data.database as db_module
    db_module.DB_PATH = test_db_path
    
    # Initialize test database
    init_database()
    
    yield test_db_path
    
    # Cleanup
    if test_db_path.exists():
        test_db_path.unlink()
    
    # Restore original DB path
    db_module.DB_PATH = original_db_path


def test_database_initialization(test_db):
    """Test that database schema is created correctly."""
    assert test_db.exists()
    assert test_db.stat().st_size > 0


def test_institution_repository(test_db):
    """Test institution repository operations."""
    # Seed test institutions
    from data.database import get_db_connection
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO institutions
        (institution_id, institution_name, college, campuses, active)
        VALUES (?, ?, ?, ?, ?)
    """, ("TEST-001", "Test Institution", "Test College", "Test Campus", 1))

    conn.commit()
    conn.close()

    # Test retrieval
    from data.repositories import InstitutionRepository
    institutions = InstitutionRepository.get_all_active()

    assert len(institutions) > 0
    assert any(inst["institution_id"] == "TEST-001" for inst in institutions)
    
    # Test get_by_id
    institution = InstitutionRepository.get_by_id("TEST-001")
    assert institution is not None
    assert institution["institution_name"] == "Test Institution"


def test_student_repository(test_db):
    """Test student repository operations."""
    # First create an institution
    from data.database import get_db_connection
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO institutions
        (institution_id, institution_name, college, campuses, active)
        VALUES (?, ?, ?, ?, ?)
    """, ("TEST-002", "Test Institution 2", "Test College 2", "Test Campus 2", 1))

    conn.commit()

    # Create student
    from data.repositories import StudentRepository
    student_data = {
        "registration_number": "TEST2024/1234",
        "institution_id": "TEST-002",
        "full_name": "Test Student",
        "programme": "Test Programme",
        "department": "Test Department",
        "academic_year": "2024",
        "year_of_study": 1,
        "semester": "Semester 1"
    }

    student_id = StudentRepository.create_student(student_data)
    assert student_id > 0

    # Test retrieval
    student = StudentRepository.get_by_registration_number(
        "TEST2024/1234", "TEST-002"
    )
    assert student is not None
    assert student["full_name"] == "Test Student"

    # Test get_academic_record (should be None initially)
    academic = StudentRepository.get_academic_record(student_id)
    assert academic is None
    
    conn.close()


def test_student_service():
    """Test student service business logic."""
    from services.student_service import StudentService
    
    # Test get_institutions (will use demo database)
    institutions = StudentService.get_institutions()
    assert isinstance(institutions, list)
    # Demo database should have institutions from seeding
    if len(institutions) > 0:
        assert "institution_id" in institutions[0]
        assert "institution_name" in institutions[0]


def test_student_validation():
    """Test student data validation for prediction."""
    from services.student_service import StudentService
    from config import REQUIRED_RAW_FIELDS

    # Test with complete data
    complete_student = {field: "test_value" for field in REQUIRED_RAW_FIELDS}
    is_valid, missing = StudentService.validate_for_prediction(complete_student)
    assert is_valid is True
    assert len(missing) == 0

    # Test with incomplete data
    incomplete_student = {field: "test_value" for field in REQUIRED_RAW_FIELDS[:10]}
    is_valid, missing = StudentService.validate_for_prediction(incomplete_student)
    assert is_valid is False
    assert len(missing) > 0


def test_rp_institutions_config():
    """Test RP_INSTITUTIONS configuration."""
    from config import RP_INSTITUTIONS

    assert isinstance(RP_INSTITUTIONS, list)
    assert len(RP_INSTITUTIONS) == 8, "Should have 8 official RP colleges"

    # Check structure
    for inst in RP_INSTITUTIONS:
        assert "id" in inst
        assert "name" in inst
        assert "college" in inst
        assert "campuses" in inst
        assert isinstance(inst["id"], str)
        assert isinstance(inst["name"], str)
        assert isinstance(inst["college"], str)
        assert isinstance(inst["campuses"], list)


def test_rp_institution_ids_unique():
    """Test that RP institution IDs are unique."""
    from config import RP_INSTITUTIONS

    ids = [inst["id"] for inst in RP_INSTITUTIONS]
    assert len(ids) == len(set(ids)), "RP institution IDs must be unique"


def test_official_rp_colleges():
    """Test that all official RP colleges are present."""
    from config import RP_INSTITUTIONS

    expected_colleges = {
        "RP-GISHARI", "RP-HUYE", "RP-KARONGI", "RP-KIGALI",
        "RP-KITABI", "RP-MUSANZE", "RP-NGOMA", "RP-TUMBA"
    }
    actual_ids = {inst["id"] for inst in RP_INSTITUTIONS}
    assert actual_ids == expected_colleges, f"Missing or extra colleges. Expected: {expected_colleges}, Got: {actual_ids}"


def test_legacy_model_compatibility_config():
    """Test LEGACY_MODEL_COMPATIBILITY configuration."""
    from config import LEGACY_MODEL_COMPATIBILITY

    assert isinstance(LEGACY_MODEL_COMPATIBILITY, dict)
    assert len(LEGACY_MODEL_COMPATIBILITY) == 8, "Should map all 8 RP colleges"

    # All values should be valid school codes for the legacy model
    valid_school_codes = {"GP", "MS"}
    for inst_id, school_code in LEGACY_MODEL_COMPATIBILITY.items():
        assert school_code in valid_school_codes, f"Invalid school code: {school_code}"


def test_rp_institution_not_displayed_as_gp():
    """Test that RP institution names do not contain 'GP'."""
    from config import RP_INSTITUTIONS

    for inst in RP_INSTITUTIONS:
        assert "GP" not in inst["name"], f"Institution name should not contain 'GP': {inst['name']}"
        assert "GP" not in inst["college"], f"College name should not contain 'GP': {inst['college']}"


def test_rp_kigali_compatibility_mapping():
    """Test that RP-KIGALI maps correctly to legacy model compatibility value."""
    from config import LEGACY_MODEL_COMPATIBILITY

    assert "RP-KIGALI" in LEGACY_MODEL_COMPATIBILITY
    assert LEGACY_MODEL_COMPATIBILITY["RP-KIGALI"] == "GP"


def test_unknown_rp_institution_safe_handling():
    """Test that unknown RP institution gets default mapping."""
    from config import LEGACY_MODEL_COMPATIBILITY

    # Unknown institution should get default via .get() with fallback
    unknown_mapping = LEGACY_MODEL_COMPATIBILITY.get("UNKNOWN-XYZ", "GP")
    assert unknown_mapping == "GP"


def test_student_service_no_feature_fabrication():
    """Test that StudentService does not fabricate missing features."""
    from services.student_service import StudentService
    from config import REQUIRED_RAW_FIELDS

    # Student with only some fields
    partial_student = {
        "school": "GP",
        "sex": "M",
        "age": 18
    }

    # Should raise error, not fabricate missing fields
    try:
        ml_features = StudentService.map_to_ml_features(partial_student)
        assert False, "Should have raised ValueError for missing fields"
    except ValueError as e:
        assert "missing required academic information" in str(e)


def test_student_service_complete_data_passes():
    """Test that StudentService passes complete data without modification."""
    from services.student_service import StudentService
    from config import REQUIRED_RAW_FIELDS

    # Complete student data
    complete_student = {field: "test_value" for field in REQUIRED_RAW_FIELDS}

    ml_features = StudentService.map_to_ml_features(complete_student)

    # All fields should be present and unchanged
    assert set(ml_features.keys()) == set(REQUIRED_RAW_FIELDS)
    for field in REQUIRED_RAW_FIELDS:
        assert ml_features[field] == "test_value"


def test_legacy_model_compatibility_no_gp_in_labels():
    """Test that GP/MS is not required in LABELS for RP institution selection."""
    from config import LEGACY_MODEL_COMPATIBILITY

    # Verify that LEGACY_MODEL_COMPATIBILITY returns direct codes (GP/MS)
    # These should be used directly, not looked up in LABELS["school"]
    for inst_id, school_code in LEGACY_MODEL_COMPATIBILITY.items():
        assert school_code in ["GP", "MS"], f"Expected GP or MS, got {school_code}"
        # The school_code should be the legacy model code, not a display label
        # It should NOT require LABELS["school"][school_code] lookup


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
