# CAT2 Assessment — Rubric Mapping

This maps every line item on the official CAT2 marking rubric (Ngoma
College, ITLPA701, /30 marks) to concrete evidence in this project.

## Apply Data Preprocessing — 30% (9 marks)

| Assessment Criteria | Marks | Project Evidence |
|---|---|---|
| Environment properly configured / functionalities specified | 1.5 | `requirements.txt`; project problem/objectives stated in README §1-3 and this document |
| Environment is configured | 1.5 | `pip install -r requirements.txt` installs pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit, joblib — the exact set actually used |
| Data properly acquired from the given data source | 3 | `data/raw/student-mat.csv` — real UCI Student Performance dataset, source cited in README §4; acquisition documented in `ACADEMIC_REPORT.md` §3.2 |
| Data appropriately pre-processed | 3 | `src/data_preprocessing.py` (cleaning) + `ColumnTransformer` in `src/train_model.py` (imputation, scaling, encoding) — verified: 0 missing values, 0 duplicates, grades clipped to [0,20] |

## Apply Deep Learning Algorithms — 50% (15 marks)

| Assessment Criteria | Marks | Project Evidence |
|---|---|---|
| Data features properly engineered | 3 | `src/feature_engineering.py` — 6 documented derived features (`average_prior_grade`, `grade_trend`, `total_study_support`, `parental_education`, `social_alcohol_index`, `attendance_risk`), each with how/why/effect justification |
| Model features engineered | 3 | Full feature set (20 numeric + 18 categorical) assembled in `src/train_model.py::prepare_dataset()`, feature importance analysed in `models/model_metadata.json` and shown on the app's Dashboard page |
| AI approach and model appropriately selected and justified | 1.5 | `ACADEMIC_REPORT.md` §3.6 — 3 candidate models compared via 5-fold CV; Gradient Boosting selected on CV F1 (0.946), justification documented |
| Important model parameters selected and explained | 1.5 | `GridSearchCV` over `n_estimators`, `max_depth`, `learning_rate` in `src/train_model.py::tune_best_model()`; best params documented in README §12 and `models/model_metadata.json` |
| Selected application correctly implemented and tested | 6 | Full working pipeline (`train_model.py`, `predict.py`, `app.py`) actually executed in this environment (not pseudocode); 10 manual tests documented in `TESTING.md` plus a 25-assertion automated `pytest` suite (`tests/`), including a real bug found and fixed |

## Apply Model Evaluation Techniques — 20% (6 marks)

| Assessment Criteria | Marks | Project Evidence |
|---|---|---|
| Appropriate evaluation metrics selected | 1 | Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC-AUC — all appropriate for binary classification |
| Evaluation correctly implemented, results clearly interpreted | 2 | `src/evaluate_model.py` reproduces metrics independently; interpretation in README §12 and `ACADEMIC_REPORT.md` §4.4 (false-positive/false-negative discussion) |
| Model parameters adjusted based on evaluation results | 1 | Baseline (untuned) vs. tuned comparison in README §12 shows a measured F1 improvement (0.913 → 0.922) from `GridSearchCV` tuning |
| Model/configuration correctly saved for reproducibility | 1 | `models/model.pkl` (joblib), `models/model_metadata.json`, `models/category_values.json` — reloaded and independently re-verified in `src/evaluate_model.py` |
| Application deployed/demonstrated through a web interface | 1 | `app.py` — 4-page Streamlit application (`streamlit run app.py`) |

**Total mapped: 30 / 30 marks**

## Additional requirement: Responsible Use

> "Discuss responsible use: Identify at least one limitation or risk and explain how it can be reduced."

Addressed in: README §13-14, `MODEL_CARD.md`, `ACADEMIC_REPORT.md` §5.2,
and the dedicated "⚠️ Responsible Use" page in `app.py`, covering bias,
privacy, data security, incorrect predictions, over-reliance on AI, and
fairness — each with a specific, actionable mitigation. The fairness
claim specifically is backed by a real, runnable subgroup audit
(`src/fairness_audit.py` → `reports/fairness_audit.md`), not just a
stated intention.
