"""
database.py
-----------
Database abstraction layer for RP Student Success System.
Uses SQLite for demo/prototype - can be replaced with PostgreSQL/official RP database.

IMPORTANT: This is a DEMO database with synthetic data.
NOT connected to real RP production systems.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from config import get_logger

logger = get_logger(__name__)

# Database path - in data/ directory
DB_PATH = Path(__file__).resolve().parents[2] / "data" / "rp_demo.db"


def get_db_connection():
    """
    Get a database connection.
    Returns a connection that should be closed by the caller.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable dict-like access
    return conn


def init_database():
    """
    Initialize the database schema.
    Creates tables if they don't exist.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Institutions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS institutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                institution_id TEXT UNIQUE NOT NULL,
                institution_name TEXT NOT NULL,
                college TEXT NOT NULL,
                campuses TEXT,
                active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Students table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                registration_number TEXT UNIQUE NOT NULL,
                institution_id TEXT NOT NULL,
                full_name TEXT NOT NULL,
                programme TEXT NOT NULL,
                department TEXT NOT NULL,
                academic_year TEXT NOT NULL,
                year_of_study INTEGER NOT NULL,
                semester TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (institution_id) REFERENCES institutions(institution_id)
            )
        """)
        
        # Student academic records table - maps to ML model features
        # All fields from REQUIRED_RAW_FIELDS plus G3 for reference
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS student_academic_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                school TEXT,
                sex TEXT,
                age INTEGER,
                address TEXT,
                famsize TEXT,
                Pstatus TEXT,
                Medu INTEGER,
                Fedu INTEGER,
                Mjob TEXT,
                Fjob TEXT,
                reason TEXT,
                guardian TEXT,
                traveltime INTEGER,
                studytime INTEGER,
                failures INTEGER,
                schoolsup TEXT,
                famsup TEXT,
                paid TEXT,
                activities TEXT,
                nursery TEXT,
                higher TEXT,
                internet TEXT,
                romantic TEXT,
                famrel INTEGER,
                freetime INTEGER,
                goout INTEGER,
                Dalc INTEGER,
                Walc INTEGER,
                health INTEGER,
                absences INTEGER,
                G1 INTEGER,
                G2 INTEGER,
                G3 INTEGER,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(id),
                UNIQUE(student_id)
            )
        """)
        
        # Interventions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interventions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                risk_level TEXT NOT NULL,
                risk_reason TEXT,
                intervention TEXT NOT NULL,
                assigned_lecturer TEXT,
                status TEXT DEFAULT 'Open',
                follow_up_date DATE,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students(id)
            )
        """)
        
        conn.commit()
        logger.info("Database schema initialized successfully")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to initialize database: {e}")
        raise
    finally:
        conn.close()


def database_exists() -> bool:
    """Check if the database file exists."""
    return DB_PATH.exists()


if __name__ == "__main__":
    # Initialize database when run directly
    init_database()
    print(f"Database initialized at: {DB_PATH}")
