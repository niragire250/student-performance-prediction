"""
feature_engineering.py
------------------------
Creates the prediction target and a small set of meaningful derived
features from the cleaned Student Performance dataset.

For every engineered feature we document:
  1. HOW it is created
  2. WHY it is useful
  3. HOW it may affect the prediction

Target definition
------------------
G3 (final grade, 0-20) is converted into a binary PASS/FAIL label using the
grading convention used in the source study and in most Portuguese/African
secondary-school systems: a final mark of 10/20 or higher is a PASS.

    pass_fail = 1 if G3 >= 10 else 0

This is NOT leakage: G3 is the ground truth we are trying to predict, and it
is removed from the feature matrix immediately after the label is created.
G1 and G2 (grades from the two earlier evaluation periods) are kept as
FEATURES, not because they equal the target, but because they represent
real information a school already has *before* the final exam. Their
strong correlation with G3 is expected and is discussed explicitly in the
Model Evaluation / Limitations sections of the report -- it is a genuine
"early formative assessment predicts final outcome" relationship, not a
copy of the target.
"""

import pandas as pd

from config import PASS_THRESHOLD, TARGET_COL, TARGET_RAW_COL, get_logger

logger = get_logger(__name__)

# Local aliases kept for readability within this module / backward compatibility
TARGET_BINARY = TARGET_COL


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived (engineered) columns to the dataframe. Returns a new df."""
    df = df.copy()

    # 1) average_prior_grade: mean of the two earlier-period grades (G1, G2)
    #    WHY: summarises a student's grade trend in a single number, which is
    #    often a stronger, less noisy signal than either grade alone.
    #    EFFECT: strongly predictive of the final outcome; also used later to
    #    build the "early warning" (G1/G2-free) comparison model.
    df["average_prior_grade"] = (df["G1"] + df["G2"]) / 2

    # 2) grade_trend: change between period 1 and period 2 grades
    #    WHY: captures whether a student is improving or declining over
    #    time, information a single snapshot grade cannot provide.
    #    EFFECT: a negative trend may flag at-risk students even if their
    #    absolute grades are still average.
    df["grade_trend"] = df["G2"] - df["G1"]

    # 3) total_study_support: count of "yes" answers among school/family
    #    support-related binary columns (schoolsup, famsup, paid classes).
    #    WHY: aggregates several small, individually-weak signals about the
    #    support a student receives into one interpretable score (0-3).
    #    EFFECT: higher support scores are expected to correlate with a
    #    higher pass probability.
    support_cols = ["schoolsup", "famsup", "paid"]
    df["total_study_support"] = (df[support_cols] == "yes").sum(axis=1)

    # 4) parental_education: average of mother's and father's education
    #    level (Medu, Fedu are already numeric 0-4 in the raw data).
    #    WHY: combines two correlated socio-economic indicators into one
    #    feature, reducing redundancy/multicollinearity for linear models.
    #    EFFECT: proxies household educational environment.
    df["parental_education"] = (df["Medu"] + df["Fedu"]) / 2

    # 5) social_alcohol_index: combined weekday + weekend alcohol
    #    consumption score (Dalc + Walc, both 1-5 scales).
    #    WHY: the two raw columns are highly correlated; combining them into
    #    one 2-10 scale reduces noise while keeping the signal.
    #    EFFECT: expected to correlate negatively with academic performance.
    df["social_alcohol_index"] = df["Dalc"] + df["Walc"]

    # 6) attendance_risk: absences bucketed into a simple risk flag
    #    WHY: raw absence counts are highly skewed (a few extreme outliers);
    #    a threshold-based flag is a more robust, interpretable signal for
    #    a linear model and for the Streamlit dashboard.
    #    EFFECT: students with >10 absences are flagged as higher risk.
    df["attendance_risk"] = (df["absences"] > 10).astype(int)

    return df


def create_target(df: pd.DataFrame,
                   threshold: int = PASS_THRESHOLD) -> pd.DataFrame:
    """
    Create the binary PASS/FAIL target from G3 and drop the raw G3 column
    from the returned dataframe (the caller decides whether to keep a copy
    of G3 separately for reporting purposes).
    """
    df = df.copy()
    df[TARGET_BINARY] = (df[TARGET_RAW_COL] >= threshold).astype(int)
    return df


if __name__ == "__main__":
    from data_preprocessing import load_data, clean_data

    raw = load_data()
    clean = clean_data(raw)
    fe = engineer_features(clean)
    fe = create_target(fe)
    logger.info("\n" + str(fe[["G1", "G2", "average_prior_grade", "grade_trend",
                                "total_study_support", "parental_education",
                                "social_alcohol_index", "attendance_risk",
                                "G3", "pass_fail"]].head()))
    logger.info("Class balance:\n" + str(fe["pass_fail"].value_counts(normalize=True)))
