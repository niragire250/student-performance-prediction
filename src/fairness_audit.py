"""
fairness_audit.py
-------------------
Checks whether the final model's performance differs across demographic
subgroups (sex, address type, school). This turns the "fairness" item in
the Responsible AI discussion from a stated intention into measured
evidence, computed on the same held-out test set used for the headline
metrics.

This is a diagnostic, not a fix: if a meaningful gap is found, the
appropriate response is documented in README.md / MODEL_CARD.md (collect
more representative data, recalibrate per-group thresholds, or restrict
deployment) rather than silently patched here.

Run: python src/fairness_audit.py
"""

import json

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from config import (FAIRNESS_AUDIT_COLUMNS, FAIRNESS_REPORT_JSON,
                     FAIRNESS_REPORT_MD, MODEL_PATH, RANDOM_STATE,
                     TEST_SIZE, get_logger)
from train_model import prepare_dataset

logger = get_logger(__name__)

MIN_GROUP_SIZE = 10  # subgroups smaller than this are reported but flagged
                      # as too small for a reliable per-group estimate


def audit_subgroup(y_true, y_pred, group_labels) -> pd.DataFrame:
    """Compute accuracy/precision/recall/F1 per distinct value of a
    demographic column, on the same test predictions used for the
    headline metrics."""
    rows = []
    df = pd.DataFrame({"y_true": y_true.values, "y_pred": y_pred, "group": group_labels.values})
    for value, sub in df.groupby("group"):
        n = len(sub)
        row = {
            "group_value": value,
            "n": n,
            "pass_rate_actual": round(sub["y_true"].mean() * 100, 1),
            "accuracy": round(accuracy_score(sub["y_true"], sub["y_pred"]) * 100, 1),
            "small_sample_warning": n < MIN_GROUP_SIZE,
        }
        # precision/recall/F1 are undefined if a subgroup has only one class
        # present in y_true or y_pred; guard against that rather than crash.
        if sub["y_true"].nunique() > 1 or sub["y_pred"].nunique() > 1:
            row["precision"] = round(precision_score(sub["y_true"], sub["y_pred"], zero_division=0) * 100, 1)
            row["recall"] = round(recall_score(sub["y_true"], sub["y_pred"], zero_division=0) * 100, 1)
            row["f1_score"] = round(f1_score(sub["y_true"], sub["y_pred"], zero_division=0) * 100, 1)
        else:
            row["precision"] = row["recall"] = row["f1_score"] = None
        rows.append(row)
    return pd.DataFrame(rows).sort_values("group_value").reset_index(drop=True)


def run_audit():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Run 'python src/train_model.py' first.")
    model = joblib.load(MODEL_PATH)

    X, y, full_df = prepare_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    # the demographic columns for the SAME test rows, by index alignment
    test_demographics = full_df.loc[X_test.index, FAIRNESS_AUDIT_COLUMNS]

    y_pred = model.predict(X_test)

    overall = {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 1),
        "f1_score": round(f1_score(y_test, y_pred) * 100, 1),
        "n_test": len(y_test),
    }
    logger.info(f"Overall test-set accuracy: {overall['accuracy']}% (n={overall['n_test']})")

    results = {}
    md_lines = ["# Fairness Audit\n",
                "Subgroup performance of the final tuned model on the held-out "
                f"test set (n={overall['n_test']}, overall accuracy "
                f"{overall['accuracy']}%).\n",
                "**Caution:** several subgroups below are small; treat percentage "
                "gaps for groups flagged `small_sample_warning` as indicative, "
                "not conclusive.\n"]

    for col in FAIRNESS_AUDIT_COLUMNS:
        table = audit_subgroup(y_test, y_pred, test_demographics[col])
        results[col] = table.to_dict(orient="records")
        logger.info(f"\nSubgroup performance by '{col}':\n{table.to_string(index=False)}")

        md_lines.append(f"\n## By `{col}`\n")
        md_lines.append("| Group | n | Actual pass rate | Accuracy | Precision | Recall | F1 | Note |")
        md_lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in table.iterrows():
            note = "small sample" if r["small_sample_warning"] else ""
            prec = f"{r['precision']}%" if r["precision"] is not None else "n/a"
            rec = f"{r['recall']}%" if r["recall"] is not None else "n/a"
            f1 = f"{r['f1_score']}%" if r["f1_score"] is not None else "n/a"
            md_lines.append(f"| {r['group_value']} | {r['n']} | {r['pass_rate_actual']}% | "
                             f"{r['accuracy']}% | {prec} | {rec} | {f1} | {note} |")

        accs = table[~table["small_sample_warning"]]["accuracy"]
        if len(accs) > 1:
            gap = accs.max() - accs.min()
            md_lines.append(f"\nLargest accuracy gap across reliably-sized `{col}` groups: "
                             f"**{gap:.1f} percentage points**.")

    md_lines.append(
        "\n## Interpretation\n\n"
        "This audit uses a single 79-student test split, so per-subgroup "
        "sample sizes are small and these figures should be treated as a "
        "first check, not a certified fairness guarantee. Before any real "
        "deployment, this audit should be repeated on a larger, more "
        "current dataset, ideally with repeated cross-validation rather "
        "than a single split, so subgroup estimates are less sensitive to "
        "which particular students happened to land in the test set."
    )

    FAIRNESS_REPORT_JSON.write_text(json.dumps(
        {"overall": overall, "by_group": results}, indent=2, default=str))
    FAIRNESS_REPORT_MD.write_text("\n".join(md_lines))
    logger.info(f"Saved fairness audit -> {FAIRNESS_REPORT_JSON}")
    logger.info(f"Saved fairness audit -> {FAIRNESS_REPORT_MD}")

    return overall, results


if __name__ == "__main__":
    run_audit()
