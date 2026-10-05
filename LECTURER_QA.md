# Anticipated Lecturer Questions & Answers

## Problem & Dataset

**Q1. Why did you choose this problem?**
Because it's a genuine, high-value use case for schools — spotting at-risk
students early enough to actually help them — and there's a real,
well-documented public dataset to build it on.

**Q2. Why did you choose this dataset?**
It's real, peer-reviewed (Cortez & Silva, 2008), from the UCI Machine
Learning Repository, contains no missing values, and has both
demographic/behavioural and academic features — everything needed to
build a meaningful classification problem.

**Q3. Is your target variable G3 leaking into your features?**
No. G3 (final grade) is the target; it's dropped from the feature matrix
right after `pass_fail` is created. G1 and G2 are *earlier-period* grades
— real information a school has before the final result — not the target
itself. I discuss this explicitly and even measure how much accuracy
depends on them (early-warning comparison, README §12).

**Q4. Why binary classification (PASS/FAIL) instead of predicting the exact grade?**
A PASS/FAIL flag is more directly actionable for a teacher than a raw
number, and it produces cleaner, more interpretable evaluation metrics
(precision/recall/confusion matrix) for a decision-support use case.

## Preprocessing

**Q5. What is preprocessing, and why is it necessary?**
Preprocessing prepares raw data for a model: handling missing values,
encoding text into numbers, and scaling numeric ranges so no feature
dominates just because of its scale.

**Q6. How do you handle missing values?**
`SimpleImputer` (median for numeric columns, most-frequent for
categorical) — inside the pipeline, so it's fit only on training data,
never on test data.

**Q7. Why did you split the data into train and test sets?**
To evaluate the model on data it never saw during training — otherwise
reported performance would be optimistic and misleading.

**Q8. What is data leakage, and how did you prevent it?**
Data leakage is when information that wouldn't be available at prediction
time (including, most subtly, information about the test set) influences
training. I prevent it by (a) fitting all preprocessing only on the
training fold via a scikit-learn `Pipeline`/`ColumnTransformer`, and
(b) never deriving features from G3 itself.

## Modelling

**Q9. Why did you choose Gradient Boosting?**
It had the highest cross-validated F1-score (0.946) among the three
models compared, and it handles non-linear relationships well without
heavy manual feature engineering — appropriate for this size of tabular
dataset.

**Q10. What is overfitting?**
When a model learns the training data too specifically (including its
noise) and performs much worse on new, unseen data.

**Q11. What is underfitting?**
When a model is too simple to capture the real patterns in the data, so
it performs poorly even on the training data.

**Q12. How did you prevent overfitting here?**
Cross-validation during model selection/tuning, a held-out test set for
final evaluation, and `GridSearchCV` selected a shallow `max_depth=2` for
the final model — the search itself favoured a simpler, less
overfitting-prone tree depth given the small dataset.

**Q13. What hyperparameters did you tune, and how?**
`n_estimators`, `max_depth`, `learning_rate` for Gradient Boosting, via
`GridSearchCV` with 5-fold cross-validation, optimising F1-score.

## Evaluation

**Q14. What is accuracy?**
The percentage of all predictions that are correct.

**Q15. What is precision?**
Of everyone the model predicted PASS, the percentage who actually pass.

**Q16. What is recall?**
Of everyone who actually passes, the percentage the model correctly
identifies.

**Q17. What is F1-score, and why use it here?**
The harmonic mean of precision and recall. I use it because the classes
are imbalanced (67% pass / 33% fail), so accuracy alone could be
misleading.

**Q18. What does your confusion matrix show?**
24 correctly predicted FAILs, 47 correctly predicted PASSes, 2 actual
FAILs wrongly predicted PASS, 6 actual PASSes wrongly predicted FAIL — out
of 79 test students.

**Q19. Which type of error matters most here, and why?**
False PASS predictions for students who actually fail (only 2 cases) —
because that's the student who would be missed and denied support. My
model's high precision (95.9%) specifically minimises this error type.

## Deployment & Responsible AI

**Q20. Why did you use Streamlit?**
It turns a Python script into an interactive web app with minimal code,
which is ideal for quickly demonstrating an ML model without building a
separate frontend.

**Q21. How do you ensure reproducibility?**
Fixed `random_state=42` everywhere, a single saved `Pipeline` object
(`model.pkl`) bundling preprocessing and model together, and metadata
(`model_metadata.json`) recording exact parameters and metrics — anyone
re-running `train_model.py` gets the same numbers.

**Q22. What are the limitations of your system?**
Small, dated (2008), geographically narrow dataset; heavy reliance on
interim grades for its best accuracy (accuracy drops from ~90% to ~67%
without them); moderate class imbalance. All discussed in README §13.

**Q23. What responsible-AI issues does this raise, and how are they addressed?**
Bias (data reflects a specific time/place), privacy (sensitive student
data), over-reliance (staff treating the output as certain) — each
addressed with a concrete mitigation in the app's "Responsible Use" page
and README §14. The core safeguard: the system is explicitly framed as
decision-support, never a decision-maker.
