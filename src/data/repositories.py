"""
repositories.py
---------------
Data access layer for student records.
Provides repository pattern for database operations.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime

from .database import get_db_connection, init_database
from config import get_logger

logger = get_logger(__name__)


class InstitutionRepository:
    """Repository for institution/campus data."""
    
    @staticmethod
    def get_all_active() -> List[Dict[str, Any]]:
        """Get all active institutions."""
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT institution_id, institution_name, college, campuses
                FROM institutions
                WHERE active = 1
                ORDER BY institution_name
            """)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    @staticmethod
    def get_by_id(institution_id: str) -> Optional[Dict[str, Any]]:
        """Get institution by ID."""
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT * FROM institutions 
                WHERE institution_id = ? AND active = 1
            """, (institution_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


class StudentRepository:
    """Repository for student records."""
    
    @staticmethod
    def get_by_registration_number(registration_number: str, 
                                   institution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get student by registration number and institution.
        
        Args:
            registration_number: Student's registration number
            institution_id: Institution ID
            
        Returns:
            Student dict with basic info, or None if not found
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT s.*, i.institution_name, i.college, i.campuses
                FROM students s
                JOIN institutions i ON s.institution_id = i.institution_id
                WHERE s.registration_number = ? AND s.institution_id = ?
            """, (registration_number, institution_id))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
    
    @staticmethod
    def get_academic_record(student_id: int) -> Optional[Dict[str, Any]]:
        """
        Get academic record for a student.
        
        Args:
            student_id: Internal student database ID
            
        Returns:
            Academic record dict with ML model features, or None if not found
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT * FROM student_academic_records 
                WHERE student_id = ?
            """, (student_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
    
    @staticmethod
    def get_full_student_record(registration_number: str, 
                                institution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get complete student record including academic data.
        
        Args:
            registration_number: Student's registration number
            institution_id: Institution ID
            
        Returns:
            Complete student dict with academic features, or None if not found
        """
        # First get basic student info
        student = StudentRepository.get_by_registration_number(
            registration_number, institution_id
        )
        if not student:
            return None
        
        # Get academic record
        academic = StudentRepository.get_academic_record(student['id'])
        
        if academic:
            # Merge student and academic data
            result = {**student, **academic}
            # Remove internal IDs from result
            result.pop('id', None)
            result.pop('student_id', None)
            return result
        
        return student
    
    @staticmethod
    def get_all_by_institution(institution_id: str) -> List[Dict[str, Any]]:
        """Get all students for an institution."""
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT s.*, i.institution_name, i.college, i.campuses
                FROM students s
                JOIN institutions i ON s.institution_id = i.institution_id
                WHERE s.institution_id = ?
                ORDER BY s.full_name
            """, (institution_id,))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    @staticmethod
    def create_student(student_data: Dict[str, Any]) -> int:
        """
        Create a new student record.
        
        Args:
            student_data: Dict with student fields
            
        Returns:
            The new student ID
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO students (
                    registration_number, institution_id, full_name, programme,
                    department, academic_year, year_of_study, semester
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                student_data['registration_number'],
                student_data['institution_id'],
                student_data['full_name'],
                student_data['programme'],
                student_data['department'],
                student_data['academic_year'],
                student_data['year_of_study'],
                student_data['semester']
            ))
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to create student: {e}")
            raise
        finally:
            conn.close()
    
    @staticmethod
    def create_academic_record(academic_data: Dict[str, Any]) -> int:
        """
        Create academic record for a student.
        
        Args:
            academic_data: Dict with academic fields (must include student_id)
            
        Returns:
            The new academic record ID
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            # Build dynamic insert statement
            fields = list(academic_data.keys())
            placeholders = ', '.join(['?'] * len(fields))
            field_names = ', '.join(fields)
            
            cursor.execute(f"""
                INSERT INTO student_academic_records ({field_names})
                VALUES ({placeholders})
            """, list(academic_data.values()))
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to create academic record: {e}")
            raise
        finally:
            conn.close()


class InterventionRepository:
    """Repository for intervention tracking."""
    
    @staticmethod
    def create(intervention_data: Dict[str, Any]) -> int:
        """Create a new intervention record."""
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO interventions (
                    student_id, risk_level, risk_reason, intervention,
                    assigned_lecturer, status, follow_up_date, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                intervention_data['student_id'],
                intervention_data['risk_level'],
                intervention_data.get('risk_reason'),
                intervention_data['intervention'],
                intervention_data.get('assigned_lecturer'),
                intervention_data.get('status', 'Open'),
                intervention_data.get('follow_up_date'),
                intervention_data.get('notes')
            ))
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to create intervention: {e}")
            raise
        finally:
            conn.close()
    
    @staticmethod
    def get_by_student(student_id: int) -> List[Dict[str, Any]]:
        """Get all interventions for a student."""
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT * FROM interventions 
                WHERE student_id = ?
                ORDER BY created_at DESC
            """, (student_id,))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    @staticmethod
    def update_status(intervention_id: int, status: str) -> bool:
        """Update intervention status."""
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                UPDATE interventions 
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (status, intervention_id))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to update intervention status: {e}")
            raise
        finally:
            conn.close()
