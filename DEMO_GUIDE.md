# Demonstration Guide (5-10 minutes)

Practical run-through for your lecturer demonstration. Keep it conversational —
you built this, so explain it in your own words using the prompts below.

## Before you start
```bash
pip install -r requirements.txt
cd src && python train_model.py     # confirms training runs end-to-end
cd ..
streamlit run app.py
```
Have a code editor open to `src/train_model.py` and `src/feature_engineering.py`,
and a terminal ready, alongside the browser tab with the Streamlit app.

## Suggested script (~8 minutes)

**1. Introduce the problem (30s)**
> "Teachers usually only see that a student is struggling once final
> grades are in. I built a system that predicts, from information
> available earlier in the term, whether a student is likely to pass or
> fail — so support can be offered in time."

**2. Explain the dataset (45s)**
> "I used the real UCI Student Performance dataset — 395 students from
> two Portuguese secondary schools, 33 real attributes: grades, study
> time, family background, attendance. It's a well-known, peer-reviewed
> academic dataset, not synthetic data." (Show `data/raw/student-mat.csv`.)

**3. Show preprocessing (45s)**
> Open `src/data_preprocessing.py`. "I check for missing values and
> duplicates — this dataset has zero of both — and clip any invalid grade
> values. The real encoding and scaling happens inside a scikit-learn
> `ColumnTransformer`, so training and prediction always use identical
> logic."

**4. Show feature engineering (45s)**
> Open `src/feature_engineering.py`. "I engineered six features, for
> example `grade_trend` — the change between a student's first and second
> period grades — which captures whether they're improving or declining,
> something no single raw column shows."

**5. Explain model selection (45s)**
> "I compared three algorithms — Logistic Regression, Random Forest and
> Gradient Boosting — with 5-fold cross-validation. Gradient Boosting won
> on F1-score, so I tuned it with GridSearchCV." (Show the printed
> comparison table from `train_model.py`.)

**6. Show training (30s)**
> Run `python src/train_model.py` live (takes ~15 seconds) or show the
> terminal output you already have. Point out the tuning step and the
> "did tuning improve F1?" line.

**7. Show evaluation metrics (60s)**
> "On 79 held-out test students, the final model reaches 89.9% accuracy
> and 92.2% F1. The confusion matrix shows only 2 students who actually
> failed were wrongly predicted to pass — the error type that matters most
> here, since that's the student who'd be missed." (Show
> `reports/figures/06_confusion_matrix.png`.)

**8. Open Streamlit (15s)**
> Switch to the browser tab already running the app.

**9. Enter sample student information (60s)**
> Fill the Predict Performance form with realistic values (or reuse the
> sample in `src/predict.py`'s `__main__` block).

**10. Generate prediction (15s)**
> Click "🔮 Predict Performance."

**11. Explain the result (30s)**
> "The system shows PASS or FAIL with a confidence percentage — never
> presented as a certainty, always a probability."

**12. Explain limitations and responsible AI (45s)**
> Open the "⚠️ Responsible Use" page. "The model is trained on a small,
> 2008 dataset from two schools, so it needs retraining on local data
> before real use. I also show that if you remove the interim grades, the
> model's accuracy drops from 90% to 67% — so I'm transparent that this
> only works well once some grades already exist. It's a decision-support
> tool, never a replacement for a teacher's judgement."

## Simple explanations to memorise

- **Accuracy** = "how often the model is right overall."
- **Precision** = "when the model says PASS, how often it's actually true."
- **Recall** = "of all students who actually pass, how many the model catches."
- **F1-score** = "a balance between precision and recall, useful when classes are imbalanced."
- **Confusion matrix** = "a 2x2 table of correct vs. wrong predictions, split by type of error."
- **Overfitting** = "the model memorises the training data instead of learning general patterns, so it does well in training but poorly on new data."
- **Data leakage** = "accidentally letting information from the answer sneak into the input features."
- **Pipeline** = "chains preprocessing and the model together so the exact same steps run every time, with no risk of mismatch."
