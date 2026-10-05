"""
test_pipeline.py
------------------
Tests for data integrity, feature engineering correctness, and the
reproducibility of the saved model's evaluation metrics. These check the
*claims* made in README.md are actually true of the current model.pkl,
not just that the code runs without crashing.

Run:  pytest tests/ -v
"""

import json

import pytest

from config import METADATA_PATH, MODEL_PATH, PASS_THRESHOLD


# --------------------------------------------------------------------------
# Data integrity
# --------------------------------------------------------------------------
def test_raw_data_has_no_missing_values():
    from data_preprocessing import load_data
    df = load_data()
    assert df.isnull().sum().sum() == 0


def test_raw_data_has_no_duplicates():
    from data_preprocessing import load_data
    df = load_data()
    assert df.duplicated().sum() == 0


def test_raw_data_shape_matches_uci_documentation():
    from data_preprocessing import load_data
    df = load_data()
    assert df.shape == (395, 33)


def test_grades_are_within_documented_range():
    from data_preprocessing import clean_data, load_data
    df = clean_data(load_data())
    for col in ["G1", "G2", "G3"]:
        assert df[col].between(0, 20).all()


# --------------------------------------------------------------------------
# Feature engineering correctness
# --------------------------------------------------------------------------
def test_engineered_features_present():
    from data_preprocessing import clean_data, load_data
    from feature_engineering import engineer_features
    df = engineer_features(clean_data(load_data()))
    expected = {"average_prior_grade", "grade_trend", "total_study_support",
                "parental_education", "social_alcohol_index", "attendance_risk"}
    assert expected.issubset(df.columns)


def test_average_prior_grade_is_mean_of_g1_g2():
    from data_preprocessing import clean_data, load_data
    from feature_engineering import engineer_features
    df = engineer_features(clean_data(load_data()))
    assert (df["average_prior_grade"] == (df["G1"] + df["G2"]) / 2).all()


def test_target_creation_uses_documented_threshold():
    from data_preprocessing import clean_data, load_data
    from feature_engineering import create_target, engineer_features
    df = create_target(engineer_features(clean_data(load_data())))
    # every PASS must have G3 >= threshold, every FAIL must have G3 < threshold
    assert (df.loc[df["pass_fail"] == 1, "G3"] >= PASS_THRESHOLD).all()
    assert (df.loc[df["pass_fail"] == 0, "G3"] < PASS_THRESHOLD).all()


def test_target_is_not_derived_from_itself_leaking():
    """G3 must not appear among the model's input features — it is the
    source of the label, not a predictor."""
    from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES
    assert "G3" not in NUMERIC_FEATURES
    assert "G3" not in CATEGORICAL_FEATURES


# --------------------------------------------------------------------------
# Saved model / reproducibility
# --------------------------------------------------------------------------
pytestmark_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="No trained model found — run 'python src/train_model.py' first.",
)


@pytestmark_model
def test_metadata_matches_saved_model_structure():
    import joblib
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    n_features_declared = len(meta["numeric_features"]) + len(meta["categorical_features"])
    # the fitted ColumnTransformer should have been built over exactly this many raw columns
    n_cols_seen = len(model.named_steps["prep"].feature_names_in_)
    assert n_features_declared == n_cols_seen


@pytestmark_model
def test_saved_metrics_are_reproducible():
    """Re-evaluating the saved model on a freshly re-split test set (same
    random_state) must reproduce the metrics recorded in model_metadata.json
    — this is what makes the numbers in README.md trustworthy rather than
    a one-off, unverifiable claim."""
    import joblib
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from config import RANDOM_STATE, TEST_SIZE
    from train_model import prepare_dataset

    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        meta = json.load(f)

    X, y, _ = prepare_dataset()
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    y_pred = model.predict(X_test)

    assert accuracy_score(y_test, y_pred) == pytest.approx(meta["final_metrics"]["accuracy"], abs=1e-6)
    assert f1_score(y_test, y_pred) == pytest.approx(meta["final_metrics"]["f1_score"], abs=1e-6)


@pytestmark_model
def test_final_model_beats_or_matches_untuned_baseline():
    """Guards against a regression where tuning accidentally makes things
    worse without anyone noticing."""
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    assert meta["final_metrics"]["f1_score"] >= meta["baseline_metrics"]["f1_score"]
