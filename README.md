# 🎓 RP Student Success & Early Warning System

A machine-learning decision-support tool for academic early warning at Rwanda Polytechnic, built with scikit-learn and Streamlit.

**Evolution**: This is an evolution of the original Student Performance Prediction System, enhanced with RP-specific features including student portal, database integration, and early warning capabilities.

---

## 1. Project Overview

This project applies supervised machine learning to an academic early-warning problem: identifying, ahead of time, which students are at risk of failing so that teachers and support staff can intervene. It covers the full ML lifecycle — data acquisition, cleaning, EDA, feature engineering, model selection, training, hyperparameter tuning, evaluation, and deployment through an interactive Streamlit web application.

### RP-Specific Features

- **Student Portal**: Predict performance using Registration Number lookup
- **Official RP Institutions**: 8 Rwanda Polytechnic colleges (Gishari, Huye, Karongi, Kigali, Kitabi, Musanze, Ngoma, Tumba)
- **Database Integration**: SQLite demo database with student records
- **Session State Management**: Persistent student selection across navigation
- **Higher-Education Context**: Terminology adapted for RP academic environment

### Model Context

**IMPORTANT**: The current prototype uses a **legacy model** trained on Portuguese secondary-school data (UCI Student Performance dataset, 2008). RP institution data is mapped for prototype compatibility via a dedicated compatibility layer. 

**Production deployment for Rwanda Polytechnic requires**:
- Validation and/or retraining using representative RP student data
- RP-specific feature engineering
- Cultural and educational context alignment

The application architecture is RP-ready, but prediction quality for real RP students has not been validated with an RP-trained model.

### Architecture Highlights

- **Database Layer**: SQLite demo database with replaceable architecture for production RP database
- **Institution Management**: Support for 8 official RP colleges with campus structure
- **Service Layer**: Clean separation between UI, business logic, and data access
- **Feature Mapping Layer**: Centralized schema and mapper for RP-to-model feature translation
- **Demo Data**: Synthetic RP student records for testing (clearly marked as DEMO/SYNTHETIC)

## 2. Problem Statement

Teachers often only discover that a student is struggling once final grades
are already in — too late for meaningful intervention. This project asks:
*can we predict, from information available during the term (attendance,
study habits, family background, early grades), whether a student is on
track to pass or at risk of failing?*

## 3. Objectives

- Acquire and clean a real, publicly available student-performance dataset.
- Engineer meaningful, non-leaking features from the raw data.
- Train and fairly compare multiple classification algorithms.
- Tune and evaluate the best model using appropriate metrics.
- Package the trained model into a usable, explainable web application.
- Discuss the system's limitations and responsible-use safeguards.

## 4. Dataset

| | |
|---|---|
| **Name** | Student Performance Data Set (Math course) |
| **Source** | UCI Machine Learning Repository |
| **URL** | https://archive.ics.uci.edu/dataset/320/student+performance |
| **Citation** | P. Cortez and A. Silva, *Using Data Mining to Predict Secondary School Student Performance*, Proceedings of 5th FUBUTEC 2008, pp. 5-12, Porto, Portugal, EUROSIS, 2008 |
| **Rows** | 395 students |
| **Columns** | 33 raw attributes |
| **Missing values** | 0 |
| **Duplicate rows** | 0 |
| **Target** | Final grade `G3` (0-20), converted to binary `pass_fail` (PASS if G3 ≥ 10) |
| **Class balance** | 265 PASS (67%) / 130 FAIL (33%) |

The dataset combines demographic (age, address, family size), social
(alcohol use, going out, relationship status), and academic (study time,
absences, past failures, period grades G1/G2) attributes collected via
school reports and student questionnaires from two Portuguese secondary
schools (Gabriel Pereira and Mousinho da Silveira).

**Numerical variables:** age, Medu, Fedu, traveltime, studytime, failures,
famrel, freetime, goout, Dalc, Walc, health, absences, G1, G2, G3.

**Categorical variables:** school, sex, address, famsize, Pstatus, Mjob,
Fjob, reason, guardian, schoolsup, famsup, paid, activities, nursery,
higher, internet, romantic.

## 5. Features

The final model uses **20 numeric** and **18 categorical** input features —
33 original attributes plus 5 engineered features (see
`src/feature_engineering.py` for full documentation of each):

- `average_prior_grade` — mean of G1 and G2
- `grade_trend` — G2 − G1 (improving/declining signal)
- `total_study_support` — count of school/family/paid support received
- `parental_education` — average of mother's and father's education level
- `social_alcohol_index` — combined weekday + weekend alcohol score
- `attendance_risk` — binary flag for students with >10 absences

## 6. Technologies Used

Python 3, pandas, NumPy, scikit-learn (Pipeline, ColumnTransformer,
SimpleImputer, StandardScaler, OneHotEncoder, GridSearchCV), matplotlib,
seaborn, Streamlit, joblib, plotly, SQLite (demo database).

## 7. Project Architecture

### Training Pipeline

```
Raw CSV (UCI dataset)
    ↓
Data Cleaning (duplicates, invalid values, whitespace)
    ↓
Exploratory Data Analysis (distributions, correlation, boxplots)
    ↓
Feature Engineering (5 derived features + PASS/FAIL target)
    ↓
Train/Test Split (80/20, stratified, random_state=42)
    ↓
Preprocessing (ColumnTransformer: impute + scale + one-hot encode)
    ↓
Baseline Model Comparison (Logistic Regression, Random Forest, Gradient Boosting)
    ↓
Hyperparameter Tuning (GridSearchCV on best baseline)
    ↓
Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
    ↓
Model Selection & Saving (joblib: model.pkl + metadata.json)
    ↓
Streamlit Deployment
```

### Prediction Architecture (RP System)

```
Streamlit UI
    ↓
Student Portal (Registration Number Lookup OR Manual Input)
    ↓
Student Service (Business Logic)
    ↓
Student Repository (Data Access)
    ↓
SQLite Demo Database (Replaceable with RP API/Database)
    ↓
Feature Mapping (Database → ML Features)
    ↓
Feature Engineering (Same as training)
    ↓
Trained Model (GradientBoosting)
    ↓
Prediction + Risk Level + Recommendations
```

### Database Schema

**institutions**: RP campus information
- institution_id, institution_name, campus, active

**students**: Student records
- registration_number, institution_id, full_name, programme, department, academic_year, year_of_study, semester

**student_academic_records**: Academic data (maps to ML features)
- All REQUIRED_RAW_FIELDS (32 fields) from the model

**interventions**: Intervention tracking
- student_id, risk_level, risk_reason, intervention, assigned_lecturer, status, follow_up_date, notes

### Prediction (inference) pipeline:

```
User Input (Streamlit form)
    ↓
Streamlit Interface
    ↓
Input Validation
    ↓
Feature Engineering (identical function as training)
    ↓
Preprocessing Pipeline (identical fitted ColumnTransformer)
    ↓
Trained ML Model (Gradient Boosting)
    ↓
Prediction
    ↓
Result + Probability + Plain-Language Interpretation
```

Using one saved `Pipeline` object (preprocessing + model together) for both
training and prediction is what guarantees there is no train/serve
mismatch.

## 8. Project File Structure

```
student-performance-prediction/
│
├── data/
│   ├── raw/
│   │   └── student-mat.csv          # original UCI dataset (unmodified)
│   ├── processed/
│   │   └── student-mat-processed.csv
│   └── rp_demo.db                   # SQLite demo database (100 synthetic students)
│
├── models/
│   ├── model.pkl                    # trained pipeline (preprocessing + classifier)
│   ├── model_metadata.json          # metrics, best params, feature lists
│   ├── category_values.json         # valid category values for the UI
│   └── baseline_comparison.json     # cross-validated baseline comparison
│
├── reports/
│   ├── figures/                     # EDA + evaluation charts (PNG)
│   ├── baseline_comparison.csv
│   ├── fairness_audit.json          # subgroup performance, machine-readable
│   └── fairness_audit.md            # subgroup performance, human-readable
│
├── src/
│   ├── config.py                    # single source of truth: paths, constants, hyperparameter grids
│   ├── data_preprocessing.py        # load, inspect, clean
│   ├── feature_engineering.py       # engineered features + target
│   ├── eda.py                       # exploratory data analysis charts
│   ├── train_model.py               # full training pipeline
│   ├── evaluate_model.py            # standalone evaluation / verification
│   ├── fairness_audit.py            # subgroup (sex/address/school) performance audit
│   ├── predict.py                   # single-record & batch inference used by app.py
│   ├── data/                        # NEW: Database abstraction layer
│   │   ├── __init__.py
│   │   ├── database.py              # SQLite database schema and connection
│   │   ├── repositories.py          # Data access layer (Student, Institution, Intervention)
│   │   └── seed.py                 # Demo data seeding script
│   └── services/                    # NEW: Business logic layer
│       ├── __init__.py
│       └── student_service.py       # Student lookup and validation logic
│
├── tests/
│   ├── conftest.py                  # shared pytest fixtures
│   ├── test_predict.py              # prediction-pipeline tests (17 tests)
│   ├── test_pipeline.py             # data-integrity & reproducibility tests (11 tests)
│   └── test_database.py             # NEW: database and repository tests (5 tests)
│
├── app.py                           # Streamlit web application (10 pages)
├── requirements.txt
├── requirements-dev.txt             # adds pytest, for running tests/
├── README.md
├── MODEL_CARD.md                    # one-page model summary (Mitchell et al. 2019 format)
├── ACADEMIC_REPORT.md
├── TESTING.md
├── RUBRIC_MAPPING.md
├── DEMO_GUIDE.md
├── LECTURER_QA.md
└── .gitignore
```

### Engineering practices applied

- **Single source of truth for constants** (`src/config.py`): `random_state`,
  the PASS/FAIL threshold, file paths and hyperparameter search grids are
  defined once and imported everywhere, so they can never silently drift
  between training, evaluation and prediction.
- **`logging` instead of `print()`** throughout `src/`, with a consistent
  timestamped format — easier to redirect, silence, or inspect than ad hoc
  print statements.
- **Automated tests** (`tests/`, pytest) formalise what was originally
  manual, ad hoc verification into repeatable, CI-friendly assertions.

## 9. Installation Instructions

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies (app + training)
pip install -r requirements.txt

# 2b. Optional — to also run the automated test suite:
pip install -r requirements-dev.txt
```

## 10. How to Train the Model

```bash
cd src
python train_model.py
```

This will print the full baseline comparison, tuning results and final
test-set metrics, generate EDA and evaluation figures (via `eda.py`, run
separately or beforehand — see below), and save `model.pkl`,
`model_metadata.json` and `category_values.json` into `models/`.

To (re)generate the EDA charts:

```bash
cd src
python eda.py
```

## 11. How to Initialize Demo Database

```bash
# Initialize database and seed demo data
python -c "import sys; sys.path.insert(0, 'src'); from data.seed import seed_all; seed_all()"
```

This creates `data/rp_demo.db` with:
- 4 RP institutions (Kigali, Ngoma, Rulindo, Kibuye campuses)
- 100 synthetic student records with academic data
- 30 intervention records

**IMPORTANT**: This is DEMO DATA - not real RP student records.

## 12. How to Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

### Available Pages

1. **Home** - Project overview and disclaimer
2. **Student Portal** - NEW: Predict using Registration Number or Manual Input
3. **Dashboard** - KPIs and visualizations
4. **Predict Performance** - Manual prediction form
5. **Early Warning** - Risk identification
6. **Student Profile** - Individual student analysis
7. **What-If Analysis** - Scenario exploration
8. **Batch Prediction** - CSV upload for multiple students
9. **Model Dashboard** - Model metrics and feature importance
10. **Fairness Audit** - Subgroup performance analysis
11. **Responsible Use** - Limitations and ethical considerations

## 13. How to Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_predict.py -v
pytest tests/test_database.py -v
```

Test coverage: 33 tests (11 pipeline + 17 prediction + 5 database)

## 14. RP Production Integration Plan

The current system uses a SQLite demo database. To integrate with official RP systems:

### Option 1: Replace with PostgreSQL

1. Update `src/data/database.py` to use `psycopg2` instead of `sqlite3`
2. Update connection string to RP PostgreSQL instance
3. Ensure schema matches RP student information system
4. Update field mappings in `StudentService.map_to_ml_features()`

### Option 2: RP API Integration

1. Create `src/data/rp_api_client.py` to call RP REST API
2. Implement `StudentRepository` methods to use API instead of database
3. Add authentication (API keys, OAuth, or RP SSO)
4. Cache API responses for performance

### Security Requirements

- Use environment variables for credentials
- Implement HTTPS for all API calls
- Add role-based access control
- Implement audit logging
- Follow RP data protection policies

## 15. Limitations

- **Demo Data**: Current database uses synthetic data, not real RP students. All records are marked as DEMO/SYNTHETIC.
- **No Authentication**: Prototype lacks RP SSO integration. Authentication is not implemented in this prototype.
- **Legacy Model**: The ML model was trained on Portuguese secondary school data (2008). RP institution data is mapped for prototype compatibility via `LEGACY_MODEL_COMPATIBILITY` configuration. Production deployment for Rwanda Polytechnic requires validation and/or retraining using representative RP student data to ensure cultural and educational context alignment.
- **RP Model Not Available**: An RP-trained model does not yet exist. The architecture is ready for RP model deployment, but prediction quality for real RP students has not been validated.
- **Feature Mapping**: Current features are based on secondary-school dataset. Higher-education specific features (GPA, credits, semester performance) are not yet implemented.
- **No Production Database**: Uses SQLite for demo. Production requires official RP database/API integration.

To independently re-verify the saved model's metrics:

```bash
cd src
python evaluate_model.py
```

## 16. How to Run the Streamlit App

```bash
# from the project root (after training the model at least once)
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

### Application Pages

The upgraded Streamlit application includes 11 professional pages:

1. **🏠 Home** - Project overview, purpose, and disclaimer
2. **👨‍🎓 Student Portal** - NEW: Predict using Registration Number lookup or Manual Input
3. **📊 Dashboard** - Professional KPI cards showing total students, pass/fail rates, average performance, and high-risk students, with interactive charts and demographic breakdowns
4. **🔮 Predict Performance** - Single student prediction with risk level (Low/Medium/High), confidence score, key influencing factors, and intervention recommendations
5. **⚠️ Early Warning** - Identify at-risk students based on configurable thresholds (absences, failures, study time, grades) with risk scores and intervention suggestions
6. **👤 Student Profile** - Detailed view of individual student academic information, demographics, support systems, social factors, and prediction with recommendations
7. **🔬 What-If Analysis** - Explore how changing student factors (study time, grades, support) affects predicted outcomes with before/after comparison
8. **📋 Batch Prediction** - Upload CSV files to predict multiple students at once with validation, summary statistics, and CSV download
9. **📈 Model Dashboard** - Comprehensive model metrics (accuracy, precision, recall, F1, ROC-AUC), confusion matrix, baseline comparison, early-warning metrics, and feature importance
10. **⚖️ Fairness Audit** - Performance analysis across demographic subgroups (sex, address, school) with detailed metrics and interpretation
11. **⚠️ Responsible Use** - Limitations, bias, privacy, data security, and mitigation strategies

## 17. Running the Automated Test Suite

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

27 assertions across two files: `tests/test_predict.py` (16 tests covering valid input,
boundary values, missing/empty/unseen-category error handling, model loading,
risk level calculation, batch prediction, feature importance, and intervention recommendations)
and `tests/test_pipeline.py` (11 tests covering data integrity — no missing values,
no duplicates, documented shape; feature-engineering correctness; and,
importantly, that the metrics reported in this README are **actually
reproducible** by re-evaluating the saved `model.pkl`, not just numbers
pasted into a document). All 27 checks pass against the current
`models/model.pkl`.

## 11c. Fairness Audit

```bash
cd src && python fairness_audit.py
```

Computes accuracy/precision/recall/F1 separately by `sex`, `address`
(urban/rural) and `school` on the same held-out test set used for the
headline metrics, and writes `reports/fairness_audit.md` /
`reports/fairness_audit.json`. Headline result: accuracy varies by about
4 points across sex and 3 points across address type — see
`reports/fairness_audit.md` for full tables and caveats about small
subgroup sizes.

## 12. Model Evaluation & Results

> These are the **actual** results obtained by running `train_model.py` on
> the real UCI dataset in this project (random_state=42, 80/20 stratified
> split, 316 train / 79 test students). Re-running the script will
> reproduce these numbers exactly.

**Baseline model comparison** (5-fold stratified cross-validation on the training set):

| Model | CV Accuracy | CV F1 | CV Precision | CV Recall |
|---|---|---|---|---|
| Gradient Boosting | 92.7% | 94.6% | 94.4% | 94.8% |
| Random Forest | 92.7% | 94.5% | 96.2% | 92.9% |
| Logistic Regression | 91.8% | 93.9% | 93.9% | 93.9% |

**Gradient Boosting** was selected for hyperparameter tuning (highest CV F1).

**Hyperparameter tuning** (GridSearchCV, 5-fold CV, scoring = F1):
Best parameters: `n_estimators=100, max_depth=2, learning_rate=0.1`
Best CV F1: **0.958**

**Final tuned model — held-out test set (79 students):**

| Metric | Untuned Baseline | Tuned Final Model |
|---|---|---|
| Accuracy | 88.6% | **89.9%** |
| Precision | 94.0% | **95.9%** |
| Recall | 88.7% | 88.7% |
| F1-score | 91.3% | **92.2%** |
| ROC-AUC | 93.8% | **94.3%** |

Tuning improved F1-score from 0.913 to 0.922 — a genuine, measured
improvement, not an assumed one.

**Confusion Matrix (test set):**

| | Predicted FAIL | Predicted PASS |
|---|---|---|
| **Actual FAIL** | 24 | 2 |
| **Actual PASS** | 6 | 47 |

**Interpretation:**
- **Accuracy (89.9%)** — the model's overall prediction is correct for 9 out of 10 students.
- **Precision (95.9%)** — when the model predicts PASS, it is right 96% of the time.
- **Recall (88.7%)** — the model correctly identifies 89% of students who actually pass.
- **Confusion matrix** — only 2 students who actually failed were missed as false PASS predictions (the most consequential error type, since these students would not be flagged for support); 6 students who actually passed were flagged as at-risk unnecessarily (a lower-cost error — extra support offered where it wasn't strictly needed).

**Supplementary "early-warning" analysis** (same model family, retrained
**without** G1/G2 — i.e. usable earlier in the term, before any exam grade
exists):

| | With G1/G2 (full model) | Without G1/G2 (early-warning) |
|---|---|---|
| Accuracy | 89.9% | 67.1% |
| F1-score | 92.2% | 78.7% |
| ROC-AUC | 94.3% | 64.8% |

This is an honest, measured trade-off: predicting earlier is possible but
notably less reliable, because prior grades carry most of the predictive
signal (feature-importance analysis shows G2 alone accounts for ~82% of the
tuned model's importance). This is discussed further in the Limitations
section.

## 13. Limitations

1. **Small, dated, geographically narrow dataset** (395 students, two
   Portuguese schools, 2008) — predictions may not generalise to other
   regions, curricula, or eras. **Mitigation:** retrain on fresh, local
   data before any real deployment; treat current results as a
   proof-of-concept.
2. **Heavy reliance on G1/G2 prior grades** — the model is far less useful
   as a true "before any grade exists" early-warning tool (67% vs 90%
   accuracy). **Mitigation:** the app's dashboard reports both numbers
   transparently; a school could choose the early-warning variant if
   predicting before any grades are recorded is the priority.
3. **Moderate class imbalance** (67% pass / 33% fail) can bias a naive
   classifier toward the majority class. **Mitigation:** stratified
   splitting/cross-validation and F1 (not just accuracy) were used for
   model selection.

## 14. Responsible AI Considerations

See `app.py` → "⚠️ Responsible Use" page, `MODEL_CARD.md`, and
`RUBRIC_MAPPING.md` / `ACADEMIC_REPORT.md` for the full discussion of
bias, privacy, data security, incorrect predictions, over-reliance, and
fairness — each with a concrete mitigation.

The fairness claim is backed by a real, runnable audit rather than a
promise alone: `src/fairness_audit.py` measures accuracy/precision/recall/F1
separately by sex, address type and school on the test set (see §11c and
`reports/fairness_audit.md`). On this dataset, the largest reliable gap
found is about 4 accuracy points (by sex) — not large relative to the ~90%
overall accuracy, but the test set is small, so this is reported as a
first check, not a certified guarantee.

In short: **this system is a decision-support tool, never a
decision-maker**, and its predictions must always be combined with a
teacher's professional judgement.

## 15. New Features in Professional Upgrade

The system has been upgraded to a professional, production-ready Student Performance Analytics System with the following enhancements:

### Enhanced Prediction Capabilities
- **Risk Level Assessment**: Each prediction now includes a risk level (Low/Medium/High) based on probability thresholds
- **Feature Importance Explanation**: Displays the top 5 factors influencing each prediction
- **Intervention Recommendations**: Provides practical, context-aware recommendations based on student data and prediction results

### New Application Pages
- **Professional Dashboard**: KPI cards, interactive charts, demographic breakdowns
- **Early Warning System**: Configurable risk thresholds, risk scoring, at-risk student identification
- **Student Profile**: Comprehensive student view with academic, demographic, and social factors
- **What-If Analysis**: Interactive exploration of how factor changes affect predictions
- **Batch Prediction**: CSV upload/download for processing multiple students
- **Fairness Audit**: Dedicated page displaying subgroup performance metrics

### Engineering Improvements
- **Batch Prediction API**: `predict_batch()` function for processing multiple records
- **Modular Code**: Reusable functions for risk calculation, feature importance, and recommendations
- **Enhanced Testing**: 27 automated tests (up from 25) covering new functionality
- **Updated Dependencies**: Added plotly for enhanced visualizations

### User Experience
- **Full-Word Labels**: All interface elements use clear, non-technical language
- **Responsive Design**: Mobile-friendly layout with CSS breakpoints
- **Professional UI**: Clean sidebar navigation, consistent layout, helpful tooltips
- **Error Handling**: User-friendly error messages and input validation

## 16. Future Improvements

- Collect and retrain on current, locally representative student data.
- Enhance fairness auditing with cross-validation on larger datasets.
- Try additional algorithms (e.g. XGBoost/LightGBM) and SHAP-based explanations.
- Add authentication and per-student audit logging for real deployment.
- Extend to a regression target (predicted numeric grade) alongside PASS/FAIL.
- Add real-time prediction API endpoints for integration with other systems.
