"""
test_predict.py
-----------------
Automated tests for the prediction pipeline (predict.py). These formalise
the manual checks originally run ad hoc during development (see
TESTING.md) as real, repeatable pytest assertions.

Run:  pytest tests/ -v
(requires a trained model: run `python src/train_model.py` first)
"""

import pytest

from config import MODEL_PATH

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="No trained model found — run 'python src/train_model.py' first.",
)


def test_valid_input_returns_prediction(base_student):
    from predict import predict_single
    result = predict_single(base_student)
    assert result["prediction"] in ("PASS", "FAIL")
    assert 0 <= result["confidence"] <= 100
    assert abs(result["probability_pass"] + result["probability_fail"] - 100) < 0.2


def test_high_prior_grades_predict_pass(base_student):
    from predict import predict_single
    student = dict(base_student, G1=18, G2=19, failures=0)
    result = predict_single(student)
    assert result["prediction"] == "PASS"
    assert result["probability_pass"] > 50


def test_low_prior_grades_predict_fail(base_student):
    from predict import predict_single
    student = dict(base_student, G1=0, G2=0, failures=3)
    result = predict_single(student)
    assert result["prediction"] == "FAIL"
    assert result["probability_fail"] > 50


def test_boundary_zero_grades_does_not_crash(base_student):
    from predict import predict_single
    student = dict(base_student, G1=0, G2=0)
    result = predict_single(student)
    assert result["prediction"] in ("PASS", "FAIL")


def test_boundary_max_grades_does_not_crash(base_student):
    from predict import predict_single
    student = dict(base_student, G1=20, G2=20)
    result = predict_single(student)
    assert result["prediction"] in ("PASS", "FAIL")


def test_boundary_high_absences_does_not_crash(base_student):
    from predict import predict_single
    student = dict(base_student, absences=100)
    result = predict_single(student)
    assert result["prediction"] in ("PASS", "FAIL")


def test_missing_field_raises_clear_error(base_student):
    from predict import predict_single
    incomplete = dict(base_student)
    del incomplete["G2"]
    with pytest.raises(ValueError, match="G2"):
        predict_single(incomplete)


def test_empty_string_field_raises_clear_error(base_student):
    from predict import predict_single
    student = dict(base_student, age="")
    with pytest.raises(ValueError, match="age"):
        predict_single(student)


def test_unseen_category_value_handled_gracefully(base_student):
    """OneHotEncoder(handle_unknown='ignore') must not crash on a category
    value that did not appear during training."""
    from predict import predict_single
    student = dict(base_student, Mjob="astronaut")
    result = predict_single(student)
    assert result["prediction"] in ("PASS", "FAIL")


def test_model_loads_as_fitted_pipeline():
    import joblib
    from sklearn.pipeline import Pipeline
    model = joblib.load(MODEL_PATH)
    assert isinstance(model, Pipeline)
    # a fitted pipeline's final estimator exposes predict()
    assert hasattr(model, "predict")


def test_category_values_cover_training_data():
    """Every dropdown value the Streamlit app can offer must be a value the
    model actually saw during training (prevents the UI from offering an
    option that silently gets one-hot-encoded as 'unknown')."""
    from predict import get_category_values
    cats = get_category_values()
    assert "school" in cats
    assert set(cats["school"]) <= {"GP", "MS"}
    assert "sex" in cats and set(cats["sex"]) <= {"F", "M"}


def test_risk_level_calculation():
    """Test that risk levels are calculated correctly based on probability."""
    from predict import calculate_risk_level
    
    # High probability pass should be low risk
    assert calculate_risk_level(0.85, "PASS") == "Low"
    assert calculate_risk_level(0.75, "PASS") == "Low"
    
    # Medium probability should be medium risk
    assert calculate_risk_level(0.55, "PASS") == "Medium"
    assert calculate_risk_level(0.45, "PASS") == "Medium"
    
    # Low probability pass should be high risk
    assert calculate_risk_level(0.25, "PASS") == "High"
    assert calculate_risk_level(0.15, "PASS") == "High"
    
    # For FAIL predictions, low probability pass is high risk
    assert calculate_risk_level(0.20, "FAIL") == "High"
    assert calculate_risk_level(0.50, "FAIL") == "Medium"
    assert calculate_risk_level(0.80, "FAIL") == "Low"


def test_batch_prediction_multiple_students(base_student):
    """Test batch prediction with multiple valid student records."""
    from predict import predict_batch
    
    students = [
        dict(base_student, G1=18, G2=19, failures=0),
        dict(base_student, G1=5, G2=6, failures=2),
        dict(base_student, G1=12, G2=13, failures=0),
    ]
    
    results = predict_batch(students)
    
    assert len(results) == 3
    for result in results:
        assert result["prediction"] in ("PASS", "FAIL")
        assert "risk_level" in result
        assert result["risk_level"] in ("Low", "Medium", "High")


def test_batch_prediction_with_missing_field(base_student):
    """Test batch prediction handles missing fields gracefully."""
    from predict import predict_batch
    
    students = [
        dict(base_student, G1=18, G2=19, failures=0),
        dict(base_student, G1=5, G2=6, failures=2),  # Valid
    ]
    
    # Remove a required field from the second student
    del students[1]["age"]
    
    results = predict_batch(students)
    
    assert len(results) == 2
    assert results[0]["prediction"] in ("PASS", "FAIL")
    assert results[1]["prediction"] == "ERROR"
    assert "error" in results[1]


def test_feature_importance_returns_dict():
    """Test that get_feature_importance returns a dictionary."""
    from predict import get_feature_importance
    
    fi = get_feature_importance()
    assert isinstance(fi, dict)
    assert len(fi) > 0


def test_intervention_recommendations(base_student):
    """Test that intervention recommendations are generated."""
    from predict import get_intervention_recommendations
    
    prediction_result = {
        "prediction": "FAIL",
        "risk_level": "High",
        "probability_pass": 25.0,
        "probability_fail": 75.0,
    }
    
    recommendations = get_intervention_recommendations(base_student, prediction_result)
    
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0
    assert all(isinstance(rec, str) for rec in recommendations)


def test_app_imports_required_raw_fields():
    """Regression test: app.py must import REQUIRED_RAW_FIELDS from config."""
    import sys
    from pathlib import Path
    
    # Read app.py and check for the import
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    with open(app_path, encoding='utf-8') as f:
        app_content = f.read()
    
    # Check that REQUIRED_RAW_FIELDS is imported from config
    assert "from config import" in app_content, "app.py must import from config"
    assert "REQUIRED_RAW_FIELDS" in app_content, "app.py must import REQUIRED_RAW_FIELDS"
    
    # Verify the import statement includes both MODEL_PATH and REQUIRED_RAW_FIELDS
    assert "MODEL_PATH, REQUIRED_RAW_FIELDS" in app_content or "REQUIRED_RAW_FIELDS, MODEL_PATH" in app_content, \
        "app.py should import both MODEL_PATH and REQUIRED_RAW_FIELDS from config"
