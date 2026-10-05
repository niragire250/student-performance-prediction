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
