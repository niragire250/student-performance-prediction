"""
evaluate_model.py
-------------------
Standalone evaluation utility. Loads the saved model and metadata and
reprints the evaluation metrics, so a lecturer can independently verify the
saved results without re-running the full training pipeline.

Run: python src/evaluate_model.py
"""

import json

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from config import METADATA_PATH, MODEL_PATH, RANDOM_STATE, TEST_SIZE, get_logger
from train_model import TARGET, prepare_dataset

logger = get_logger(__name__)


def load_trained_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No trained model found. Run 'python src/train_model.py' first.")
    return joblib.load(MODEL_PATH)


def main():
    model = load_trained_model()
    with open(METADATA_PATH) as f:
        meta = json.load(f)

    X, y, _ = prepare_dataset()
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    logger.info(f"Loaded model family: {meta['best_model_family']}")
    logger.info(f"Best hyperparameters: {meta['best_params']}")
    logger.info(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    logger.info(f"Precision: {precision_score(y_test, y_pred):.3f}")
    logger.info(f"Recall   : {recall_score(y_test, y_pred):.3f}")
    logger.info(f"F1-score : {f1_score(y_test, y_pred):.3f}")
    logger.info(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.3f}")
    logger.info("Classification report:\n" +
                classification_report(y_test, y_pred, target_names=["FAIL", "PASS"]))
    cm = confusion_matrix(y_test, y_pred)
    logger.info("Confusion matrix (rows=actual, cols=predicted):\n" + np.array2string(cm))

    tn, fp, fn, tp = cm.ravel()
    logger.info("Interpretation:")
    logger.info(f"- {tp} students who actually PASSED were correctly predicted to pass.")
    logger.info(f"- {tn} students who actually FAILED were correctly predicted to fail.")
    logger.info(f"- {fp} students who actually FAILED were incorrectly predicted to PASS "
                f"(false positives -> the riskiest error: an at-risk student is missed).")
    logger.info(f"- {fn} students who actually PASSED were incorrectly predicted to FAIL "
                f"(false negatives -> a student is flagged for support unnecessarily).")
    logger.info("In a decision-support context for student welfare, false positives "
                "(missing an at-risk student, i.e. false PASS predictions) are more "
                "costly than false negatives, so recall on the FAIL class is closely "
                "monitored alongside overall accuracy.")

    return {"accuracy": accuracy_score(y_test, y_pred), "f1_score": f1_score(y_test, y_pred),
            "confusion_matrix": cm.tolist()}


if __name__ == "__main__":
    main()
