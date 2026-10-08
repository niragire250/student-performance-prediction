"""
config.py
----------
Single source of truth for paths, constants and hyperparameter search
spaces used across the project. Importing from here (instead of repeating
literals in every script) is what guarantees, for example, that
`random_state=42` and the PASS/FAIL threshold can never silently drift
between training, evaluation and prediction.
"""

import logging
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "raw" / "student-mat.csv"
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
CATEGORY_VALUES_PATH = MODELS_DIR / "category_values.json"
BASELINE_COMPARISON_CSV = REPORTS_DIR / "baseline_comparison.csv"
BASELINE_COMPARISON_JSON = MODELS_DIR / "baseline_comparison.json"
FAIRNESS_REPORT_JSON = REPORTS_DIR / "fairness_audit.json"
FAIRNESS_REPORT_MD = REPORTS_DIR / "fairness_audit.md"

for d in (PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Reproducibility & modelling constants
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

TARGET_RAW_COL = "G3"          # raw numeric final grade, 0-20
TARGET_COL = "pass_fail"       # binary classification target
PASS_THRESHOLD = 10            # G3 >= PASS_THRESHOLD -> PASS (1), else FAIL (0)

# Columns considered "sensitive" for the fairness audit (demographic
# attributes that must not, on their own, justify differential treatment).
FAIRNESS_AUDIT_COLUMNS = ["sex", "address", "school"]

# --------------------------------------------------------------------------
# RP Institution Configuration (Single Source of Truth)
# --------------------------------------------------------------------------
# Official Rwanda Polytechnic colleges and campuses
RP_INSTITUTIONS = [
    {
        "id": "RP-GISHARI",
        "name": "Rwanda Polytechnic - Gishari College",
        "college": "Gishari",
        "campuses": []
    },
    {
        "id": "RP-HUYE",
        "name": "Rwanda Polytechnic - Huye College",
        "college": "Huye",
        "campuses": []
    },
    {
        "id": "RP-KARONGI",
        "name": "Rwanda Polytechnic - Karongi College",
        "college": "Karongi",
        "campuses": []
    },
    {
        "id": "RP-KIGALI",
        "name": "Rwanda Polytechnic - Kigali College",
        "college": "Kigali",
        "campuses": ["Rutongo"]
    },
    {
        "id": "RP-KITABI",
        "name": "Rwanda Polytechnic - Kitabi College",
        "college": "Kitabi",
        "campuses": ["Rusizi"]
    },
    {
        "id": "RP-MUSANZE",
        "name": "Rwanda Polytechnic - Musanze College",
        "college": "Musanze",
        "campuses": []
    },
    {
        "id": "RP-NGOMA",
        "name": "Rwanda Polytechnic - Ngoma College",
        "college": "Ngoma",
        "campuses": []
    },
    {
        "id": "RP-TUMBA",
        "name": "Rwanda Polytechnic - Tumba College",
        "college": "Tumba",
        "campuses": []
    }
]

# --------------------------------------------------------------------------
# Model Configuration
# --------------------------------------------------------------------------
# Legacy model: Portuguese secondary school data (GP/MS)
# RP model: Not yet trained - architecture ready, awaiting RP training data
LEGACY_MODEL_PATH = MODEL_PATH
RP_MODEL_PATH = ROOT / "models" / "rp_student_success_model.pkl"

# Model availability status
RP_MODEL_AVAILABLE = RP_MODEL_PATH.exists()

# --------------------------------------------------------------------------
# Legacy Model Compatibility Layer
# --------------------------------------------------------------------------
# The legacy model was trained on Portuguese school data (GP/MS).
# This mapping is ONLY for legacy model compatibility during prototype phase.
# In production with RP-specific data, use the RP-trained model.
# DO NOT present legacy model predictions as RP-specific predictions.
LEGACY_MODEL_COMPATIBILITY = {
    "RP-GISHARI": "GP",
    "RP-HUYE": "GP",
    "RP-KARONGI": "GP",
    "RP-KIGALI": "GP",
    "RP-KITABI": "GP",
    "RP-MUSANZE": "GP",
    "RP-NGOMA": "GP",
    "RP-TUMBA": "GP"
}

# --------------------------------------------------------------------------
# Feature lists
# --------------------------------------------------------------------------
BINARY_YESNO_COLS = [
    "schoolsup", "famsup", "paid", "activities", "nursery",
    "higher", "internet", "romantic",
]
NOMINAL_COLS = ["school", "sex", "address", "famsize", "Pstatus",
                 "Mjob", "Fjob", "reason", "guardian"]

RAW_NUMERIC_COLS = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
    "G1", "G2",
]
ENGINEERED_NUMERIC_COLS = [
    "average_prior_grade", "grade_trend", "total_study_support",
    "parental_education", "social_alcohol_index",
]
NUMERIC_FEATURES = RAW_NUMERIC_COLS + ENGINEERED_NUMERIC_COLS
CATEGORICAL_FEATURES = NOMINAL_COLS + BINARY_YESNO_COLS + ["attendance_risk"]

REQUIRED_RAW_FIELDS = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime",
    "failures", "schoolsup", "famsup", "paid", "activities", "nursery",
    "higher", "internet", "romantic", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences", "G1", "G2",
]

# --------------------------------------------------------------------------
# Hyperparameter search spaces (GridSearchCV) per candidate model family
# --------------------------------------------------------------------------
PARAM_GRIDS = {
    "RandomForest": {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [4, 8, None],
        "clf__min_samples_leaf": [1, 2, 4],
    },
    "GradientBoosting": {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [2, 3, 4],
        "clf__learning_rate": [0.05, 0.1],
    },
    "LogisticRegression": {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__penalty": ["l2"],
    },
}


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with consistent formatting across the
    project. Using logging instead of print() means output includes the
    source module and level, and can be silenced/redirected centrally
    (e.g. when this code runs inside the Streamlit app)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
