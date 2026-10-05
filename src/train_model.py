"""
train_model.py
----------------
Full, reproducible training pipeline for the Student Performance
Prediction System.

Steps
-----
1. Load + clean the raw data (data_preprocessing.py)
2. Engineer features + create the PASS/FAIL target (feature_engineering.py)
3. Split X / y and train / test (stratified, random_state=42)
4. Build a scikit-learn ColumnTransformer + Pipeline (prevents data
   leakage: all imputation/scaling/encoding is *fit only on the training
   fold*, never on the full dataset)
5. Train three candidate baseline models and compare them with
   stratified cross-validation
6. Hyperparameter-tune the best baseline with GridSearchCV
7. Evaluate the tuned model on the held-out test set
8. Run a supplementary "early-warning" comparison (model trained WITHOUT
   G1/G2) to discuss the trade-off between accuracy and how early a
   prediction can be made
9. Save the final pipeline + metadata with joblib for the Streamlit app

Run:  python src/train_model.py
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                      cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (BASELINE_COMPARISON_CSV, BASELINE_COMPARISON_JSON,
                     CATEGORICAL_FEATURES, CATEGORY_VALUES_PATH, CV_FOLDS,
                     FIGURES_DIR, METADATA_PATH, MODEL_PATH, MODELS_DIR,
                     NUMERIC_FEATURES, PARAM_GRIDS, RANDOM_STATE,
                     REPORTS_DIR, TARGET_COL, TEST_SIZE, get_logger)
from data_preprocessing import clean_data, load_data
from feature_engineering import create_target, engineer_features

logger = get_logger(__name__)
TARGET = TARGET_COL  # local alias kept for readability in this module


def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    """
    ColumnTransformer used for BOTH training and prediction, guaranteeing
    identical preprocessing logic in both pipelines (no train/serve skew).
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])


def prepare_dataset():
    df = clean_data(load_data())
    df = engineer_features(df)
    df = create_target(df)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df[TARGET]
    return X, y, df


def compare_baseline_models(X_train, y_train, preprocessor) -> pd.DataFrame:
    """Cross-validate three candidate algorithms and return a comparison table."""
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "f1", "precision", "recall"]

    rows = []
    for name, model in candidates.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", model)])
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
        rows.append({
            "model": name,
            "cv_accuracy_mean": scores["test_accuracy"].mean(),
            "cv_accuracy_std": scores["test_accuracy"].std(),
            "cv_f1_mean": scores["test_f1"].mean(),
            "cv_precision_mean": scores["test_precision"].mean(),
            "cv_recall_mean": scores["test_recall"].mean(),
        })
    return pd.DataFrame(rows).sort_values("cv_f1_mean", ascending=False).reset_index(drop=True)


def tune_best_model(best_name, X_train, y_train, preprocessor):
    """Hyperparameter-tune the best baseline model with GridSearchCV.
    Model instances and search spaces are centralised in config.PARAM_GRIDS
    so the grid used here can never drift from what's documented."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    model_factory = {
        "RandomForest": lambda: RandomForestClassifier(random_state=RANDOM_STATE),
        "GradientBoosting": lambda: GradientBoostingClassifier(random_state=RANDOM_STATE),
        "LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    }
    model = model_factory[best_name]()
    param_grid = PARAM_GRIDS[best_name]

    pipe = Pipeline([("prep", preprocessor), ("clf", model)])
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, target_names=["FAIL", "PASS"]),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    return metrics


def plot_confusion_matrix(cm, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(5, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["FAIL", "PASS"], yticklabels=["FAIL", "PASS"])
    plt.title("Confusion Matrix - Final Tuned Model (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_feature_importance(model, numeric_cols, categorical_cols, path):
    """Plot feature importance if the final estimator supports it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clf = model.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return None

    ohe = model.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(ohe.get_feature_names_out(categorical_cols))
    all_names = numeric_cols + cat_names

    importances = clf.feature_importances_
    idx = np.argsort(importances)[-15:]
    plt.figure(figsize=(8, 6))
    plt.barh([all_names[i] for i in idx], importances[idx], color="#4C72B0")
    plt.title("Top 15 Feature Importances - Final Model")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return dict(zip([all_names[i] for i in idx], importances[idx].tolist()))


def early_warning_comparison(X_train, X_test, y_train, y_test, preprocessor_full):
    """
    Supplementary analysis: retrain the SAME algorithm family without G1/G2
    to quantify the accuracy cost of predicting earlier (before any exam
    grade exists). Used for the Model Improvement / Limitations discussion.
    """
    drop_cols = ["G1", "G2", "average_prior_grade", "grade_trend"]
    numeric_early = [c for c in NUMERIC_FEATURES if c not in drop_cols]
    preprocessor_early = build_preprocessor(numeric_early, CATEGORICAL_FEATURES)

    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=RANDOM_STATE)
    pipe = Pipeline([("prep", preprocessor_early), ("clf", model)])
    pipe.fit(X_train[numeric_early + CATEGORICAL_FEATURES], y_train)
    metrics = evaluate_model(pipe, X_test[numeric_early + CATEGORICAL_FEATURES], y_test)
    return metrics


def main():
    logger.info("=" * 70)
    logger.info("STUDENT PERFORMANCE PREDICTION - MODEL TRAINING PIPELINE")
    logger.info("=" * 70)

    X, y, full_df = prepare_dataset()
    logger.info(f"Dataset ready: {X.shape[0]} students, {X.shape[1]} input features.")
    logger.info(f"Class balance -> PASS: {(y==1).sum()} | FAIL: {(y==0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    logger.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    preprocessor = build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)

    # ---- Step 1: Compare baseline models ----
    logger.info(f"--- Baseline model comparison ({CV_FOLDS}-fold stratified CV on train set) ---")
    comparison = compare_baseline_models(X_train, y_train, preprocessor)
    logger.info("\n" + comparison.to_string(index=False))
    comparison.to_csv(BASELINE_COMPARISON_CSV, index=False)

    best_name = comparison.iloc[0]["model"]
    logger.info(f"Best baseline model selected for tuning: {best_name}")

    # ---- Step 2: Evaluate the untuned best baseline on the test set ----
    model_factory = {
        "LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest": lambda: RandomForestClassifier(random_state=RANDOM_STATE),
        "GradientBoosting": lambda: GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    baseline_pipe = Pipeline([("prep", preprocessor), ("clf", model_factory[best_name]())])
    baseline_pipe.fit(X_train, y_train)
    baseline_metrics = evaluate_model(baseline_pipe, X_test, y_test)
    logger.info(f"Untuned {best_name} test-set metrics: "
                f"Accuracy={baseline_metrics['accuracy']:.3f}  F1={baseline_metrics['f1_score']:.3f}")

    # ---- Step 3: Hyperparameter tuning ----
    logger.info(f"--- Hyperparameter tuning ({best_name}, GridSearchCV) ---")
    t0 = time.time()
    tuned_model, best_params, best_cv_f1 = tune_best_model(best_name, X_train, y_train, preprocessor)
    logger.info(f"Tuning finished in {time.time()-t0:.1f}s | "
                f"Best params: {best_params} | Best CV F1: {best_cv_f1:.3f}")

    # ---- Step 4: Final evaluation on the held-out test set ----
    final_metrics = evaluate_model(tuned_model, X_test, y_test)
    logger.info("--- FINAL TUNED MODEL - TEST SET EVALUATION ---")
    logger.info(f"Accuracy={final_metrics['accuracy']:.3f}  "
                f"Precision={final_metrics['precision']:.3f}  "
                f"Recall={final_metrics['recall']:.3f}  "
                f"F1={final_metrics['f1_score']:.3f}" +
                (f"  ROC-AUC={final_metrics['roc_auc']:.3f}" if "roc_auc" in final_metrics else ""))
    logger.info("Classification report:\n" + final_metrics["classification_report"])
    logger.info("Confusion matrix:\n" + str(np.array(final_metrics["confusion_matrix"])))

    improved = final_metrics["f1_score"] >= baseline_metrics["f1_score"]
    logger.info(f"Did tuning improve F1 over the untuned baseline? "
                f"{'YES' if improved else 'NO (baseline kept as reference)'} "
                f"({baseline_metrics['f1_score']:.3f} -> {final_metrics['f1_score']:.3f})")

    # ---- Step 5: Confusion matrix + feature importance plots ----
    plot_confusion_matrix(np.array(final_metrics["confusion_matrix"]),
                           FIGURES_DIR / "06_confusion_matrix.png")
    importances = plot_feature_importance(
        tuned_model, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
        FIGURES_DIR / "07_feature_importance.png")

    # ---- Step 6: Early-warning (no G1/G2) supplementary comparison ----
    logger.info("--- Supplementary analysis: early-warning model WITHOUT G1/G2 ---")
    early_metrics = early_warning_comparison(X_train, X_test, y_train, y_test, preprocessor)
    logger.info(f"Early-warning model (no prior grades) -> "
                f"Accuracy={early_metrics['accuracy']:.3f}, F1={early_metrics['f1_score']:.3f}")
    logger.info(f"Full model (with G1/G2)               -> "
                f"Accuracy={final_metrics['accuracy']:.3f}, F1={final_metrics['f1_score']:.3f}")

    # ---- Step 7: Save the final model + all metadata ----
    joblib.dump(tuned_model, MODEL_PATH)

    metadata = {
        "target": TARGET,
        "pass_threshold_on_G3": 10,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "best_model_family": best_name,
        "best_params": best_params,
        "random_state": RANDOM_STATE,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "baseline_metrics": {k: v for k, v in baseline_metrics.items()
                              if k != "confusion_matrix"},
        "final_metrics": {k: v for k, v in final_metrics.items()
                           if k != "confusion_matrix"},
        "confusion_matrix": final_metrics["confusion_matrix"],
        "early_warning_metrics": {k: v for k, v in early_metrics.items()
                                   if k != "confusion_matrix"},
        "top_feature_importances": importances,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # A small sample of raw category values, used by the Streamlit app to
    # populate dropdowns exactly as seen in training (no mismatch risk).
    category_values = {col: sorted(full_df[col].dropna().unique().tolist())
                        for col in CATEGORICAL_FEATURES}
    with open(CATEGORY_VALUES_PATH, "w") as f:
        json.dump(category_values, f, indent=2)

    comparison.to_json(BASELINE_COMPARISON_JSON, orient="records", indent=2)

    logger.info(f"Saved model      -> {MODEL_PATH}")
    logger.info(f"Saved metadata   -> {METADATA_PATH}")
    logger.info(f"Saved categories -> {CATEGORY_VALUES_PATH}")
    logger.info("Training pipeline complete.")

    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
            "model": tuned_model, "metadata": metadata}


if __name__ == "__main__":
    main()
