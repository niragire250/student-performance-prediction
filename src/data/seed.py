"""
seed.py
-------
Seeds the demo database with synthetic RP student data.

IMPORTANT: This is DEMO DATA - NOT REAL RP STUDENT RECORDS.
All data is synthetic and for demonstration purposes only.
"""

import random
from datetime import datetime, timedelta

from .database import init_database, get_db_connection
from .repositories import InstitutionRepository, StudentRepository, InterventionRepository
from config import REQUIRED_RAW_FIELDS, get_logger, RP_INSTITUTIONS

logger = get_logger(__name__)


def seed_institutions():
    """
    Seed RP institutions into the database.
    Uses official Rwanda Polytechnic colleges.
    """
    institutions = []
    for inst in RP_INSTITUTIONS:
        institutions.append({
            "institution_id": inst["id"],
            "institution_name": inst["name"],
            "college": inst["college"],
            "campuses": ",".join(inst["campuses"]) if inst["campuses"] else "",
            "active": 1
        })
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        for inst in institutions:
            cursor.execute("""
                INSERT OR IGNORE INTO institutions
                (institution_id, institution_name, college, campuses, active)
                VALUES (?, ?, ?, ?, ?)
            """, (inst["institution_id"], inst["institution_name"],
                  inst["college"], inst["campuses"], inst["active"]))
        
        conn.commit()
        logger.info(f"Seeded {len(institutions)} institutions")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to seed institutions: {e}")
        raise
    finally:
        conn.close()


def seed_demo_students(count: int = 50):
    """
    Seed demo students with synthetic academic data.
    
    Args:
        count: Number of students to create
    """
    institutions = InstitutionRepository.get_all_active()
    
    if not institutions:
        raise ValueError("No institutions found. Seed institutions first.")
    
    programmes = [
        "Information Technology",
        "Electrical Engineering",
        "Civil Engineering",
        "Agriculture Technology",
        "Hospitality Management",
        "Business Management"
    ]
    
    departments = [
        "ICT Department",
        "Engineering Department",
        "Agriculture Department",
        "Tourism Department",
        "Business Department"
    ]
    
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        for i in range(count):
            institution = random.choice(RP_INSTITUTIONS)
            year = random.randint(2023, 2024)
            reg_num = f"RP{year}/{random.randint(1000, 9999)}"

            # Create student record
            student_data = {
                "registration_number": reg_num,
                "institution_id": institution["id"],
                "full_name": f"Demo Student {i+1}",
                "programme": random.choice(programmes),
                "department": random.choice(departments),
                "academic_year": str(year),
                "year_of_study": random.randint(1, 3),
                "semester": random.choice(["Semester 1", "Semester 2"])
            }

            student_id = StudentRepository.create_student(student_data)

            # Create academic record with ML model features
            # Note: Using legacy model features for compatibility
            # In production with RP model, this would use RP-specific features
            from config import LEGACY_MODEL_COMPATIBILITY
            academic_data = {
                "student_id": student_id,
                "school": LEGACY_MODEL_COMPATIBILITY.get(institution["id"], "GP"),
                "sex": random.choice(["M", "F"]),
                "age": random.randint(15, 22),
                "address": random.choice(["U", "R"]),
                "famsize": random.choice(["GT3", "LE3"]),
                "Pstatus": random.choice(["T", "A"]),
                "Medu": random.randint(0, 4),
                "Fedu": random.randint(0, 4),
                "Mjob": random.choice(["teacher", "health", "services", "at_home", "other"]),
                "Fjob": random.choice(["teacher", "health", "services", "at_home", "other"]),
                "reason": random.choice(["home", "reputation", "course", "other"]),
                "guardian": random.choice(["mother", "father", "other"]),
                "traveltime": random.randint(1, 4),
                "studytime": random.randint(1, 4),
                "failures": random.randint(0, 3),
                "schoolsup": random.choice(["yes", "no"]),
                "famsup": random.choice(["yes", "no"]),
                "paid": random.choice(["yes", "no"]),
                "activities": random.choice(["yes", "no"]),
                "nursery": random.choice(["yes", "no"]),
                "higher": random.choice(["yes", "no"]),
                "internet": random.choice(["yes", "no"]),
                "romantic": random.choice(["yes", "no"]),
                "famrel": random.randint(1, 5),
                "freetime": random.randint(1, 5),
                "goout": random.randint(1, 5),
                "Dalc": random.randint(1, 5),
                "Walc": random.randint(1, 5),
                "health": random.randint(1, 5),
                "absences": random.randint(0, 20),
                "G1": random.randint(0, 20),
                "G2": random.randint(0, 20),
                "G3": random.randint(0, 20)
            }
            
            StudentRepository.create_academic_record(academic_data)
        
        conn.commit()
        logger.info(f"Seeded {count} demo students")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to seed students: {e}")
        raise
    finally:
        conn.close()


def seed_interventions(count: int = 20):
    """Seed demo intervention records."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get some students
        cursor.execute("SELECT id FROM students LIMIT 50")
        student_ids = [row[0] for row in cursor.fetchall()]
        
        if not student_ids:
            logger.warning("No students found. Seed students first.")
            return
        
        risk_levels = ["Low", "Medium", "High"]
        statuses = ["Open", "In Progress", "Completed", "Referred"]
        
        for i in range(count):
            student_id = random.choice(student_ids)
            
            intervention_data = {
                "student_id": student_id,
                "risk_level": random.choice(risk_levels),
                "risk_reason": random.choice([
                    "High absences",
                    "Declining grades",
                    "Low study time",
                    "Past failures"
                ]),
                "intervention": random.choice([
                    "Academic counseling session",
                    "Tutoring assignment",
                    "Attendance monitoring",
                    "Study skills workshop"
                ]),
                "assigned_lecturer": f"Lecturer {random.randint(1, 10)}",
                "status": random.choice(statuses),
                "follow_up_date": (datetime.now() + timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d"),
                "notes": "Demo intervention record"
            }
            
            InterventionRepository.create(intervention_data)
        
        conn.commit()
        logger.info(f"Seeded {count} demo interventions")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to seed interventions: {e}")
        raise
    finally:
        conn.close()


def seed_all():
    """Seed all demo data."""
    logger.info("Starting database seeding...")
    
    # Initialize database schema
    init_database()
    
    # Seed data
    seed_institutions()
    seed_demo_students(100)
    seed_interventions(30)
    
    logger.info("Database seeding completed successfully")


if __name__ == "__main__":
    seed_all()
    print("Demo database seeded successfully!")
