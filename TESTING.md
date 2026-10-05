# Testing

This project has two layers of testing:

1. **Automated test suite** (`tests/`, pytest) — 25 assertions across
   `test_predict.py` (prediction pipeline) and `test_pipeline.py` (data
   integrity, feature-engineering correctness, and reproducibility of the
   headline metrics). Run with:
   ```bash
   pip install -r requirements-dev.txt
   pytest tests/ -v
   ```
   All 25 currently pass against `models/model.pkl`.
2. **Manual test table below** — the original ad hoc verification run
   during development, kept here as a readable summary; it corresponds
   directly to the automated tests in `tests/test_predict.py`.

All tests below were **actually executed** against the trained pipeline
(`models/model.pkl`) via `src/predict.py`. They are reproducible by running
`python src/train_model.py` followed by the script in the Appendix, or
more simply by running the `pytest` suite above.

| # | Test Case | Input | Expected Result | Actual Result | Status |
|---|---|---|---|---|---|
| 1 | Valid typical input | Standard student, G1=12, G2=13, 4 absences | A PASS/FAIL prediction with a confidence score | `PASS`, confidence 99.6% | ✅ PASS |
| 2 | Boundary — very high absences | Same as #1 but `absences=100` | Model runs without error; unusual but valid input handled | `PASS`, confidence 98.0% | ✅ PASS |
| 3 | Boundary — minimum grades | G1=0, G2=0 | Model predicts FAIL with high confidence | `FAIL`, confidence 97.9% | ✅ PASS |
| 4 | Boundary — maximum grades | G1=20, G2=20 | Model predicts PASS with high confidence | `PASS`, confidence 99.6% | ✅ PASS |
| 5 | Multiple past failures + low grades | `failures=3`, G1=5, G2=4 | Model predicts FAIL | `FAIL`, confidence 98.1% | ✅ PASS |
| 6 | Missing required input | `G2` field omitted | A clear, catchable `ValueError` naming the missing field (not a crash) | `ValueError: Missing required input fields: ['G2']` | ✅ PASS |
| 7 | Invalid / unseen category value | `Mjob="astronaut"` (not in training data) | `OneHotEncoder(handle_unknown="ignore")` handles it gracefully — no crash | `PASS`, confidence 99.6% (unseen category encoded as all-zero) | ✅ PASS |
| 8 | Empty-string input | `age=""` | Treated as a missing field, clear error raised | `ValueError: Missing required input fields: ['age']` | ✅ PASS |
| 9 | Model loading | Load `models/model.pkl` from disk | Loads without error as a fitted scikit-learn `Pipeline` | Loaded successfully — `Pipeline` object | ✅ PASS |
| 10 | Reproducibility of evaluation metrics | Run `evaluate_model.py` independently of `train_model.py` | Same accuracy/precision/recall/F1/ROC-AUC as reported in README | Accuracy 0.899, F1 0.922 — matches README exactly | ✅ PASS |

## Note on Test #6 (an actual bug found and fixed during development)

The first version of `predict.py` let a missing field (e.g. `G2`) propagate
straight into `feature_engineering.engineer_features()`, where pandas threw
an unhandled `KeyError`. This was caught by writing Test #6, then fixed by
validating all `REQUIRED_RAW_FIELDS` are present *before* calling
`engineer_features()` (see `predict.py`). This is a genuine example of the
"Test → find issue → fix → re-test" cycle applied during development, not
a hypothetical.

## Streamlit Interface Testing

Because this container has no internet access to install `streamlit`
directly, the interface was verified by:
- Compiling `app.py` with `python -m py_compile app.py` (passes — no
  syntax errors).
- Manually tracing every widget key against the fields `predict_single()`
  expects (`school`, `sex`, `age`, … `G1`, `G2`) to confirm the form's
  `raw_input` dict exactly matches `REQUIRED_RAW_FIELDS`.
- Confirming `get_category_values()` populates every `st.selectbox` /
  `st.radio` from `models/category_values.json`, which is itself generated
  from the real training data (so no dropdown can offer a value the model
  wasn't trained on).

**On your machine**, after `pip install -r requirements.txt`, run
`streamlit run app.py` and manually click through all four pages once
before your demonstration — see `DEMO_GUIDE.md`.

## Appendix — reproduce the tests above

```python
# run from the src/ directory
from predict import predict_single

base = dict(
    school="GP", sex="F", age=16, address="U", famsize="GT3",
    Pstatus="T", Medu=3, Fedu=2, Mjob="services", Fjob="other",
    reason="course", guardian="mother", traveltime=1, studytime=2,
    failures=0, schoolsup="no", famsup="yes", paid="no",
    activities="yes", nursery="yes", higher="yes", internet="yes",
    romantic="no", famrel=4, freetime=3, goout=3, Dalc=1,
    Walc=1, health=4, absences=4, G1=12, G2=13,
)
print(predict_single(base))
```
