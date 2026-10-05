# Model Card: Student PASS/FAIL Predictor

Following the model card format popularised by Mitchell et al. (2019) and
adopted by Google and Hugging Face — a single, scannable reference for
what this model does, how it was built, and where it should and shouldn't
be used.

## Model Details

| | |
|---|---|
| **Model type** | Gradient Boosting Classifier (scikit-learn `GradientBoostingClassifier`) |
| **Version** | 1.0 |
| **Hyperparameters** | `n_estimators=100, max_depth=2, learning_rate=0.1` (selected via 5-fold `GridSearchCV`, optimising F1) |
| **Input** | 20 numeric + 18 categorical student attributes (38 total after preprocessing; see README.md §5) |
| **Output** | Binary label `PASS` / `FAIL` + class probability |
| **Framework** | scikit-learn, packaged as a single `Pipeline` (preprocessing + classifier), serialised with `joblib` |
| **Training date** | This project build (random_state=42, fully reproducible — see Appendix A of the academic report) |
| **License / authors** | Educational project for CAT2 Practical Assessment, ITLPA701, Ngoma College |

## Intended Use

**Primary intended use:** a decision-*support* signal that flags students
who may be at risk of failing, early enough in the term for a teacher to
offer additional support. Intended users are teachers, tutors and academic
administrators at the secondary-school level.

**Out-of-scope uses:**
- Automated or sole decision-making about a student's grading, placement,
  discipline, scholarship eligibility, or any formal outcome.
- Deployment on student populations materially different from the
  training data (different country, curriculum, age group, or era)
  without retraining and re-validation.
- Any use that treats the output as a certainty rather than a probability.

## Training Data

UCI Machine Learning Repository, *Student Performance Data Set* (Math
course): 395 students from two Portuguese secondary schools, collected via
school reports and student questionnaires (Cortez & Silva, 2008). 33 raw
attributes spanning demographics, family background, study habits,
social/lifestyle factors, and two prior-period grades (G1, G2). Full
details: README.md §4.

The binary target (`pass_fail`) is derived from the raw final grade `G3`
(0–20 scale): `PASS` if `G3 ≥ 10`, else `FAIL`. Class balance: 67% PASS /
33% FAIL.

**Known data limitations:** small sample (395 students), single country
and era (Portugal, ~2008), two specific schools only. See "Limitations"
below.

## Evaluation

Evaluated once on a held-out, stratified 20% test split (79 students) that
the model never saw during training or hyperparameter selection.

| Metric | Value |
|---|---|
| Accuracy | 89.9% |
| Precision (PASS) | 95.9% |
| Recall (PASS) | 88.7% |
| F1-score | 92.2% |
| ROC-AUC | 94.3% |

Confusion matrix: 24 true negatives, 2 false positives, 6 false negatives,
47 true positives (of 79). Full interpretation: README.md §12.

### Subgroup (fairness) evaluation

A real audit was run on the same test set across `sex`, `address`
(urban/rural), and `school`. Full tables in `reports/fairness_audit.md`;
headline findings:

| Subgroup split | Accuracy range | Gap |
|---|---|---|
| sex (F / M) | 87.5% – 91.5% | 4.0 pts |
| address (R / U) | 89.4% – 92.3% | 2.9 pts |
| school (GP / MS) | 88.7% – 100%* | *MS has only 8 test students — flagged as too small to trust |

No subgroup gap found is large relative to the overall ~90% baseline, but
the test set is small (79 students total, some subgroups under 15), so
this should be read as a first check, not a certified fairness guarantee.
See `reports/fairness_audit.md` for the full methodology and caveats.

### Early-warning trade-off

A supplementary evaluation retrained the same model family **without**
`G1`/`G2` (i.e. usable before any grade exists): accuracy drops to 67.1%
and F1 to 78.7% (from 89.9% / 92.2%). The model's strength is heavily
concentrated in prior grades — `G2` alone accounts for ~82% of total
feature importance. See README.md §12 for the full comparison.

## Limitations

1. **Small, dated, geographically narrow training set** — 395 students,
   two Portuguese schools, data collected around 2008. Results may not
   generalise to other regions, curricula, school systems, or eras.
2. **Accuracy is highly dependent on prior grades (G1/G2)** — the model is
   far less reliable as a true "before any grade exists" predictor (see
   Early-warning trade-off above).
3. **Small test set (79 students)** — point estimates for accuracy/F1/etc.
   carry real sampling uncertainty; subgroup estimates even more so.
4. **No temporal validation** — the single train/test split does not test
   whether the model generalises to a *different cohort or year* of
   students, only to held-out students from the same cohort.
5. **Correlational, not causal** — a PASS/FAIL prediction reflects
   statistical association in historical data, not a causal explanation
   of why a given student might struggle.

## Ethical Considerations

- **Bias:** socio-demographic features (family structure, parental job,
  address type) are part of the input. The subgroup audit above found no
  large gap on this dataset, but that does not guarantee fairness on a
  different population; re-audit before any new deployment.
- **Privacy:** training data includes sensitive attributes (alcohol use,
  health, family status). The Streamlit app processes input in-memory
  only and does not log or transmit student data.
- **Over-reliance:** the system is designed and must be presented as a
  decision-support tool. The application surfaces this disclaimer on
  every relevant page; it should never be the sole basis for a decision
  affecting a student.

## How to Reproduce

```bash
pip install -r requirements.txt
cd src
python train_model.py        # trains, tunes, evaluates, saves model.pkl
python evaluate_model.py     # independently re-verifies the metrics above
python fairness_audit.py     # regenerates the subgroup audit
cd .. && pytest tests/ -v    # runs the automated test suite (needs requirements-dev.txt)
```

`random_state=42` is fixed throughout, so re-running reproduces every
number in this card exactly.

## References

Mitchell, M. et al. (2019). *Model Cards for Model Reporting.* FAT* '19.
Cortez, P. and Silva, A. (2008). *Using Data Mining to Predict Secondary
School Student Performance.* Proceedings of 5th FUBUTEC, Porto, Portugal.
