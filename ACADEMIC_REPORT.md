# Academic Project Report

## Student Performance Prediction System Using Machine Learning

**Module:** Python and Fundamentals of AI (ITLPA701)
**Institution:** Rwanda Polytechnic — Ngoma College
**Assessment:** CAT2 Practical Assessment

---

## Chapter 1: Introduction

### 1.1 Background
Academic underperformance and school dropout remain persistent challenges
in secondary education. Teachers often identify struggling students only
after formal exams, when opportunities for early support have already
passed. Machine learning offers a way to detect early-warning patterns in
data schools already collect — attendance, study habits, family context,
and interim grades — before a student's final result is fixed.

### 1.2 Problem Statement
Given a set of academic, demographic, social and behavioural attributes
for a student, can a machine-learning model reliably predict whether that
student will **PASS** or **FAIL** their final assessment, early enough to
be useful for intervention?

### 1.3 Main Objective
To design, implement, evaluate and deploy a machine-learning system that
predicts student academic outcomes (PASS/FAIL) from real, publicly
available student data, and to package it as an interactive, explainable
decision-support web application.

### 1.4 Specific Objectives
1. Acquire a real, appropriately sourced dataset relevant to student performance.
2. Clean and preprocess the data using a leakage-safe pipeline.
3. Perform exploratory data analysis to understand the data and justify modelling choices.
4. Engineer meaningful, interpretable derived features.
5. Compare multiple candidate classification algorithms with proper cross-validation.
6. Tune and evaluate the selected model using appropriate metrics.
7. Interpret the results in the context of the real-world problem.
8. Save the trained pipeline so the system is fully reproducible.
9. Build a professional Streamlit interface for live demonstration.
10. Identify limitations and discuss responsible-AI safeguards.

### 1.5 Project Questions
- Which student attributes are most predictive of final academic outcome?
- How much does knowledge of interim grades (G1, G2) improve prediction
  accuracy compared to demographic/behavioural data alone?
- Can a single, reusable preprocessing pipeline guarantee consistent
  behaviour between training and live prediction?

### 1.6 Scope
The project covers the Mathematics-course subset (395 students) of the UCI
Student Performance dataset. It is a binary classification problem
(PASS/FAIL), not a multi-subject or multi-year longitudinal study.

### 1.7 Significance
The project demonstrates, end-to-end, how a school could build a
lightweight, transparent early-warning tool using open data and standard
ML tooling, while explicitly surfacing the tool's limitations so it is
used responsibly rather than blindly trusted.

---

## Chapter 2: Literature / Technical Background

### 2.1 Artificial Intelligence
Artificial Intelligence (AI) is the field of building systems that perform
tasks normally requiring human intelligence, such as recognising patterns,
making predictions, or making decisions under uncertainty.

### 2.2 Machine Learning
Machine Learning (ML) is a subfield of AI in which systems learn patterns
directly from data rather than being explicitly programmed with rules. A
model is *trained* on historical examples and then used to make
predictions on new, unseen data.

### 2.3 Supervised Learning
Supervised learning uses labelled historical data — pairs of inputs (X)
and known correct outputs (y) — to learn a function that maps inputs to
outputs. This project is a supervised learning problem: X is the student's
attributes, y is the PASS/FAIL outcome.

### 2.4 Classification vs. Regression
- **Classification** predicts a discrete category (e.g. PASS/FAIL). This
  project uses classification because the practical, actionable output a
  teacher needs is a simple risk flag.
- **Regression** predicts a continuous number (e.g. the exact grade 0-20).
  The underlying `G3` variable is numeric, so a regression formulation was
  possible; classification was chosen for interpretability and direct
  actionability (see Chapter 3 for the full justification).

### 2.5 Selected Algorithm: Gradient Boosting
Gradient Boosting builds an ensemble of shallow decision trees
sequentially, where each new tree corrects the errors of the previous
ones. It generally achieves strong performance on small-to-medium
structured/tabular datasets like this one, and naturally captures
non-linear relationships and feature interactions without heavy manual
engineering.

### 2.6 Relevant Evaluation Metrics
- **Accuracy** — proportion of all predictions that are correct.
- **Precision** — of all students predicted to pass, the proportion who actually pass.
- **Recall** — of all students who actually pass, the proportion correctly identified.
- **F1-score** — harmonic mean of precision and recall; balances both concerns.
- **Confusion Matrix** — a table of actual vs. predicted outcomes, showing all four error types.
- **ROC-AUC** — measures how well the model separates the two classes across all decision thresholds.

---

## Chapter 3: Methodology

### 3.1 Dataset
The **UCI Student Performance dataset** (Math course, `student-mat.csv`)
was used: 395 real students, 33 attributes, collected via school reports
and questionnaires at two Portuguese secondary schools (Cortez & Silva,
2008). Source:
https://archive.ics.uci.edu/dataset/320/student+performance

### 3.2 Data Acquisition
The dataset was retrieved from a verified public mirror of the official
UCI release and validated against the UCI documentation (395 rows, 33
columns, 0 missing values, 0 duplicates — all confirmed programmatically
in `src/data_preprocessing.py`).

### 3.3 Data Preprocessing
Implemented in `src/data_preprocessing.py`:
- **Loading & inspection** — confirm row/column counts and data types.
- **Duplicate removal** — `drop_duplicates()` (0 found in this dataset, but
  the step is defensive for future data refreshes).
- **Invalid value handling** — grade columns are clipped to the documented
  valid range [0, 20].
- **Whitespace normalisation** on categorical/text columns.
- **Missing-value imputation** — handled *inside* the modelling pipeline
  (`SimpleImputer`, median for numeric / most-frequent for categorical) so
  that the exact same logic that would run in production also runs during
  training, fit only on the training fold (no leakage from test data).
- **Encoding** — `OneHotEncoder(handle_unknown="ignore")` for categorical
  features.
- **Scaling** — `StandardScaler` for numeric features (needed for
  Logistic Regression; harmless for tree-based models).
- **Train/test split** — 80/20, stratified on the target, `random_state=42`
  for reproducibility.

### 3.4 Exploratory Data Analysis
Implemented in `src/eda.py`, generating five figures (`reports/figures/`):
1. Target distribution (G3 histogram + PASS/FAIL bar chart) — confirms a
   67/33 class balance, motivating the use of F1-score alongside accuracy.
2. Correlation heatmap — confirms G1/G2 are the strongest numeric
   correlates of G3, and that most other numeric features are weakly
   correlated with each other (limited multicollinearity concern).
3. Feature distributions (studytime, absences, failures, age).
4. Boxplots of studytime/absences/failures split by PASS/FAIL — visually
   confirms passing students tend to have fewer failures and (mildly)
   fewer absences.
5. Categorical bar charts — pass rate by sex and by "wants higher
   education", the latter showing a clear positive association.

### 3.5 Feature Engineering
Implemented in `src/feature_engineering.py`. Five features were derived
(see file docstrings for the full how/why/effect discussion of each):
`average_prior_grade`, `grade_trend`, `total_study_support`,
`parental_education`, `social_alcohol_index`, plus a binary
`attendance_risk` flag. None of these features use `G3` in their
computation, so none constitute target leakage.

### 3.6 Model Selection
Three candidate algorithms were compared with 5-fold stratified
cross-validation on the training set only:
- Logistic Regression (interpretable linear baseline)
- Random Forest (bagged trees, robust to outliers/non-linearity)
- Gradient Boosting (sequential boosted trees, typically strongest on
  small tabular datasets)

**Gradient Boosting** had the highest cross-validated F1-score (0.946) and
was selected for hyperparameter tuning.

### 3.7 Training & Hyperparameter Tuning
`GridSearchCV` (5-fold, scoring=F1) searched over `n_estimators`,
`max_depth`, and `learning_rate`. Best configuration:
`n_estimators=100, max_depth=2, learning_rate=0.1` (CV F1 = 0.958). Small
`max_depth` was selected by the search itself, consistent with the small
dataset size (risk of overfitting deeper trees).

### 3.8 Evaluation
The tuned model was evaluated once, on the untouched 79-student test set
(see Chapter 4 for full results).

---

## Chapter 4: Implementation and Results

### 4.1 System Architecture
See `README.md` §7 for the full training and inference architecture
diagrams. In summary: a single fitted scikit-learn `Pipeline` (preprocessing
+ classifier) is saved with `joblib` and reused, unchanged, for every
prediction made through the Streamlit app — this is what guarantees the
training and serving logic never drift apart.

### 4.1b Software Engineering Practices

Beyond the modelling itself, the codebase applies three practices typical
of production (not just notebook-level) ML code: a single `src/config.py`
module centralising every constant (random seed, PASS threshold, file
paths, hyperparameter search grids) so they cannot silently drift between
scripts; `logging` instead of `print()` throughout for consistent,
redirectable output; and an automated `pytest` suite (`tests/`, 25
assertions) that checks data integrity, feature-engineering correctness,
and — critically — that the metrics reported in this document are
actually reproducible by re-running the saved model, not just numbers
pasted into a report.

### 4.2 Streamlit Application

The application has been evolved into the **RP Student Success & Early Warning System** prototype with 11 pages:

- **Home** (overview, purpose, disclaimer)
- **Student Portal** (Registration Number lookup with session state management)
- **Dashboard** (KPIs and visualizations)
- **Predict Performance** (data-entry form → prediction + confidence, with RP Institution/College selector)
- **Early Warning** (risk identification)
- **Student Profile** (individual student analysis)
- **What-If Analysis** (scenario exploration)
- **Batch Prediction** (CSV upload for multiple students)
- **Model Dashboard** (dataset stats, metrics, confusion matrix, feature importance, early-warning comparison)
- **Fairness Audit** (subgroup performance analysis)
- **Responsible Use** (risk/mitigation table, includes legacy model context)

**Important RP Prototype Context**: The application uses a **legacy model** trained on Portuguese secondary-school data (2008). RP institution data is mapped for prototype compatibility via `LEGACY_MODEL_COMPATIBILITY` configuration. Production deployment for Rwanda Polytechnic requires validation and/or retraining using representative RP student data. The application architecture is RP-ready, but an RP-trained model does not yet exist.

*(See DEMO_GUIDE.md for a screenshot-by-screenshot walkthrough to perform during your live demonstration — this document does not embed screenshots, since they must be captured from your own running application.)*

### 4.3 Model Results
*(Full numbers, with interpretation, are documented in README.md §12 —
reproduced in summary here.)*

- Final tuned model (Gradient Boosting): **Accuracy 89.9%, Precision
  95.9%, Recall 88.7%, F1 92.2%, ROC-AUC 94.3%** on the held-out test set.
- Confusion matrix: 24 true negatives, 2 false positives, 6 false
  negatives, 47 true positives (of 79 test students).
- Tuning improved F1 from 0.913 (untuned baseline) to 0.922 (tuned) — a
  measured, not assumed, improvement.
- G2 (period-2 grade) alone accounts for ~82% of the tuned model's total
  feature importance — the single strongest predictor of final outcome.

### 4.3b Subgroup Fairness Audit

Beyond overall test-set metrics, `src/fairness_audit.py` was built to
measure whether performance differs across demographic subgroups — sex,
address type (urban/rural), and school — on the same 79-student test set.
This turns the fairness item in Chapter 5's Responsible AI discussion
from a stated intention into measured evidence. Results (full tables in
`reports/fairness_audit.md`):

| Split | Accuracy range | Gap |
|---|---|---|
| sex (F / M) | 87.5% – 91.5% | 4.0 pts |
| address (R / U) | 89.4% – 92.3% | 2.9 pts |
| school (GP / MS) | 88.7% – 100%* | *MS: only 8 test students, flagged as unreliable |

No gap found is large relative to the ~90% overall accuracy, but with a
79-student test set these should be read as a first check rather than a
certified guarantee — a limitation stated explicitly in the audit's own
output, not glossed over.

### 4.4 Interpretation
The high precision (95.9%) means the model rarely wrongly predicts PASS
for a student who will actually fail — desirable, since false PASS
predictions are the error that would cause an at-risk student to be
missed. The confusion matrix confirms this: only 2 of 26 actually-failing
students were missed. Recall (88.7%) shows the model still successfully
identifies most passing students, avoiding excessive unnecessary
interventions.

---

## Chapter 5: Conclusion and Recommendations

### 5.1 Summary
This project built a complete, reproducible ML pipeline — from raw UCI
data to a deployed Streamlit application — that predicts student PASS/FAIL
outcomes with 89.9% test-set accuracy and 92.2% F1-score, using a tuned
Gradient Boosting classifier trained on 20 numeric and 18 categorical
features (5 of which were engineered).

### 5.2 Limitations
See README.md §13 for the full discussion: dataset size/age/geography,
heavy reliance on interim grades for accuracy, and moderate class
imbalance.

### 5.3 Recommendations
- Before any real school deployment, retrain on current, local student
  data and audit performance across demographic subgroups.
- Use the early-warning (no G1/G2) variant only when a true "before any
  grade exists" prediction is required, with the understanding that its
  accuracy is meaningfully lower.
- Always pair model output with a teacher's professional judgement — never
  use it as a sole decision input.

### 5.4 Future Improvements
See README.md §15.
