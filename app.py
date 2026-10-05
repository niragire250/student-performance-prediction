"""
app.py
-------
Streamlit web application for the Student Performance Prediction System.

Run with:
    streamlit run app.py

Pages:
  1. Home                - project overview, purpose, disclaimer
  2. Predict              - data entry form + PASS/FAIL prediction
  3. Model Dashboard      - dataset info, evaluation metrics, feature importance
  4. Responsible Use      - limitations, bias, privacy, mitigation
"""

import sys
from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from predict import predict_single, get_metadata, get_category_values  # noqa: E402
from config import MODEL_PATH  # noqa: E402

st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
)


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
st.sidebar.title("🎓 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🔮 Predict Performance", "📊 Model Dashboard", "⚠️ Responsible Use"],
)

if not MODEL_PATH.exists():
    st.sidebar.error(
        "No trained model found.\n\nRun `python src/train_model.py` first, "
        "then restart this app."
    )

st.sidebar.markdown("---")
st.sidebar.caption(
    "Student Performance Prediction System · CAT2 Practical Project\n\n"
    "Module: Python and Fundamentals of AI (ITLPA701)"
)


# --------------------------------------------------------------------------
# PAGE 1: HOME
# --------------------------------------------------------------------------
if page == "🏠 Home":
    st.title("🎓 Student Performance Prediction System")
    st.markdown("#### A Machine-Learning Decision-Support Tool for Academic Early Warning")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Project Description")
        st.write(
            "This system predicts whether a secondary-school student is likely "
            "to **PASS** or **FAIL** their final mathematics assessment, based "
            "on academic, demographic, social and behavioural information "
            "collected earlier in the school year. It is built on the real, "
            "peer-reviewed **UCI Student Performance dataset** (Cortez & Silva, "
            "2008) covering 395 students from two Portuguese secondary schools."
        )

        st.subheader("Purpose")
        st.write(
            "The goal is **not** to replace a teacher's judgement, but to give "
            "teachers, tutors and academic administrators an early, "
            "data-informed signal about which students may need additional "
            "support — while there is still time to act."
        )

        st.subheader("How the System Works")
        st.markdown(
            """
            1. A staff member enters a student's information (study time,
               attendance, past grades, family background, etc.) in the
               **Predict Performance** page.
            2. The input is passed through the *same* preprocessing pipeline
               used during training (missing-value handling, scaling,
               encoding, feature engineering).
            3. A trained **Gradient Boosting classifier** estimates the
               probability that the student will pass.
            4. The system displays the prediction, a confidence score, and a
               short plain-language interpretation.
            """
        )

    with col2:
        st.info(
            "**⚠️ Important Disclaimer**\n\n"
            "This tool produces a *statistical estimate*, not a certainty and "
            "not a decision. It must never be used as the sole basis for any "
            "decision that affects a student (grading, placement, discipline, "
            "or admission). Always combine its output with a teacher's "
            "professional judgement and the student's full context."
        )
        st.metric("Students in training data", "395")
        st.metric("Source", "UCI ML Repository")
        st.metric("Task type", "Binary Classification")

    st.markdown("---")
    st.caption(
        "Dataset citation: P. Cortez and A. Silva. *Using Data Mining to "
        "Predict Secondary School Student Performance.* In A. Brito and J. "
        "Teixeira (Eds.), Proceedings of 5th FUBUTEC 2008, pp. 5-12, Porto, "
        "Portugal, EUROSIS, 2008. "
        "Source: https://archive.ics.uci.edu/dataset/320/student+performance"
    )


# --------------------------------------------------------------------------
# PAGE 2: PREDICT
# --------------------------------------------------------------------------
elif page == "🔮 Predict Performance":
    st.title("🔮 Predict Student Performance")

    if not MODEL_PATH.exists():
        st.error("Model not found. Please run `python src/train_model.py` first.")
        st.stop()

    cats = get_category_values()

    # ----------------------------------------------------------------
    # Full-word display labels for every coded / abbreviated dataset
    # value (e.g. "U" -> "Urban", "at_home" -> "Stay-at-home parent").
    # The widgets below show only the full-word labels; the dictionaries
    # are used once, at submission time, to map the person's choice back
    # to the short code the trained model actually expects — so nothing
    # about the model or its inputs changes, only what the user reads.
    # ----------------------------------------------------------------
    LABELS = {
        "school": {"Gabriel Pereira (GP)": "GP", "Mousinho da Silveira (MS)": "MS"},
        "sex": {"Female": "F", "Male": "M"},
        "address": {"Urban": "U", "Rural": "R"},
        "famsize": {"More than 3 family members": "GT3", "3 or fewer family members": "LE3"},
        "Pstatus": {"Parents living together": "T", "Parents living apart": "A"},
        "Mjob": {"Teacher": "teacher", "Health care related": "health",
                 "Civil services (administrative, police, etc.)": "services",
                 "Stay-at-home parent": "at_home", "Other": "other"},
        "Fjob": {"Teacher": "teacher", "Health care related": "health",
                 "Civil services (administrative, police, etc.)": "services",
                 "Stay-at-home parent": "at_home", "Other": "other"},
        "reason": {"Close to home": "home", "School reputation": "reputation",
                   "Course preference": "course", "Other reason": "other"},
        "guardian": {"Mother": "mother", "Father": "father", "Other guardian": "other"},
        "schoolsup": {"Yes": "yes", "No": "no"},
        "famsup": {"Yes": "yes", "No": "no"},
        "paid": {"Yes": "yes", "No": "no"},
        "activities": {"Yes": "yes", "No": "no"},
        "nursery": {"Yes": "yes", "No": "no"},
        "higher": {"Yes": "yes", "No": "no"},
        "internet": {"Yes": "yes", "No": "no"},
        "romantic": {"Yes": "yes", "No": "no"},
        "studytime": {"Less than 2 hours": 1, "2 to 5 hours": 2,
                      "5 to 10 hours": 3, "More than 10 hours": 4},
        "traveltime": {"Less than 15 minutes": 1, "15 to 30 minutes": 2,
                       "30 minutes to 1 hour": 3, "More than 1 hour": 4},
        "failures": {"None": 0, "1 past failure": 1,
                     "2 past failures": 2, "3 or more past failures": 3},
    }

    def options_for(field):
        """Full-word options for `field`. For categorical fields, limited to
        the raw values the model actually saw during training (so the UI
        can never offer a choice the model wasn't trained on). Numeric
        coded fields (studytime, traveltime, failures) aren't in
        category_values.json — their full, documented 1-4 / 0-3 domain is
        used as-is."""
        if field in cats:
            raw_present = set(cats[field])
            return [full for full, raw in LABELS[field].items() if raw in raw_present]
        return list(LABELS[field].keys())

    st.write("Enter the student's information below, then click **Predict Performance**.")

    with st.form("prediction_form"):
        st.markdown("##### 👤 Demographics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            school = st.selectbox("School", options_for("school"))
        with c2:
            sex = st.selectbox("Sex", options_for("sex"))
        with c3:
            age = st.slider("Age", 15, 22, 17)
        with c4:
            address = st.selectbox("Home address type", options_for("address"))

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            famsize = st.selectbox("Family size", options_for("famsize"))
        with c2:
            Pstatus = st.selectbox("Parents' cohabitation status", options_for("Pstatus"))
        with c3:
            Medu = st.select_slider("Mother's education (0=none, 4=higher)",
                                     options=[0, 1, 2, 3, 4], value=2)
        with c4:
            Fedu = st.select_slider("Father's education (0=none, 4=higher)",
                                     options=[0, 1, 2, 3, 4], value=2)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            Mjob = st.selectbox("Mother's job", options_for("Mjob"))
        with c2:
            Fjob = st.selectbox("Father's job", options_for("Fjob"))
        with c3:
            reason = st.selectbox("Reason chose school", options_for("reason"))
        with c4:
            guardian = st.selectbox("Guardian", options_for("guardian"))

        st.markdown("##### 📚 Academic Information")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            studytime = st.select_slider("Weekly study time",
                                          options=options_for("studytime"))
        with c2:
            failures = st.select_slider("Past class failures",
                                         options=options_for("failures"))
        with c3:
            traveltime = st.select_slider("Home-to-school travel time",
                                           options=options_for("traveltime"))
        with c4:
            absences = st.number_input("Number of absences", min_value=0, max_value=100, value=4)

        c1, c2 = st.columns(2)
        with c1:
            G1 = st.slider("Period 1 grade (G1, 0-20)", 0, 20, 12)
        with c2:
            G2 = st.slider("Period 2 grade (G2, 0-20)", 0, 20, 12)

        st.markdown("##### 🏠 Support & Activities")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            schoolsup = st.radio("Extra school support", options_for("schoolsup"), horizontal=True)
        with c2:
            famsup = st.radio("Family educational support", options_for("famsup"), horizontal=True)
        with c3:
            paid = st.radio("Extra paid classes", options_for("paid"), horizontal=True)
        with c4:
            activities = st.radio("Extracurricular activities", options_for("activities"), horizontal=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            nursery = st.radio("Attended nursery school", options_for("nursery"), horizontal=True)
        with c2:
            higher = st.radio("Wants higher education", options_for("higher"), horizontal=True)
        with c3:
            internet = st.radio("Internet access at home", options_for("internet"), horizontal=True)

        romantic = st.radio("In a romantic relationship", options_for("romantic"), horizontal=True)

        st.markdown("##### 💬 Social & Wellbeing")
        c1, c2, c3 = st.columns(3)
        with c1:
            famrel = st.slider("Quality of family relationships (1-5)", 1, 5, 4)
            freetime = st.slider("Free time after school (1-5)", 1, 5, 3)
        with c2:
            goout = st.slider("Going out with friends (1-5)", 1, 5, 3)
            health = st.slider("Current health status (1-5)", 1, 5, 4)
        with c3:
            Dalc = st.slider("Workday alcohol consumption (1-5)", 1, 5, 1)
            Walc = st.slider("Weekend alcohol consumption (1-5)", 1, 5, 1)

        submitted = st.form_submit_button("🔮 Predict Performance", use_container_width=True)

    if submitted:
        # Map every full-word selection back to the short code the trained
        # model expects (e.g. "Urban" -> "U", "5 to 10 hours" -> 3).
        raw_input = dict(
            school=LABELS["school"][school], sex=LABELS["sex"][sex], age=age,
            address=LABELS["address"][address], famsize=LABELS["famsize"][famsize],
            Pstatus=LABELS["Pstatus"][Pstatus], Medu=Medu, Fedu=Fedu,
            Mjob=LABELS["Mjob"][Mjob], Fjob=LABELS["Fjob"][Fjob],
            reason=LABELS["reason"][reason], guardian=LABELS["guardian"][guardian],
            traveltime=LABELS["traveltime"][traveltime],
            studytime=LABELS["studytime"][studytime],
            failures=LABELS["failures"][failures],
            schoolsup=LABELS["schoolsup"][schoolsup], famsup=LABELS["famsup"][famsup],
            paid=LABELS["paid"][paid], activities=LABELS["activities"][activities],
            nursery=LABELS["nursery"][nursery], higher=LABELS["higher"][higher],
            internet=LABELS["internet"][internet], romantic=LABELS["romantic"][romantic],
            famrel=famrel, freetime=freetime, goout=goout, Dalc=Dalc, Walc=Walc,
            health=health, absences=absences, G1=G1, G2=G2,
        )
        try:
            result = predict_single(raw_input)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        st.markdown("---")
        st.subheader("Prediction Result")

        col1, col2, col3 = st.columns(3)
        with col1:
            if result["prediction"] == "PASS":
                st.success(f"### Prediction: {result['prediction']} ✅")
            else:
                st.error(f"### Prediction: {result['prediction']} ⚠️")
        with col2:
            st.metric("Confidence", f"{result['confidence']}%")
        with col3:
            st.metric("P(Pass)", f"{result['probability_pass']}%")

        st.progress(int(result["probability_pass"]))

        if result["prediction"] == "PASS":
            st.write(
                f"The model estimates a **{result['probability_pass']}% probability** "
                "that this student will pass, based on the patterns learned from "
                "395 historical students with similar characteristics."
            )
        else:
            st.write(
                f"The model estimates a **{result['probability_fail']}% probability** "
                "that this student is at risk of failing. Consider reviewing "
                "study support, attendance and prior-grade trends with the student."
            )

        st.warning(
            "⚠️ This is a probabilistic estimate from a statistical model, "
            "**not a certainty and not a decision**. Please combine this "
            "result with your own professional judgement."
        )


# --------------------------------------------------------------------------
# PAGE 3: MODEL DASHBOARD
# --------------------------------------------------------------------------
elif page == "📊 Model Dashboard":
    st.title("📊 Model Information Dashboard")

    if not MODEL_PATH.exists():
        st.error("Model not found. Please run `python src/train_model.py` first.")
        st.stop()

    meta = get_metadata()

    st.subheader("Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total students", meta["n_train"] + meta["n_test"])
    c2.metric("Training samples", meta["n_train"])
    c3.metric("Testing samples", meta["n_test"])
    c4.metric("Input features", len(meta["numeric_features"]) + len(meta["categorical_features"]))

    st.subheader("Selected Model")
    st.write(f"**Algorithm:** {meta['best_model_family']}")
    st.json(meta["best_params"])

    st.subheader("Test-Set Evaluation Metrics")
    fm = meta["final_metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{fm['accuracy']*100:.1f}%")
    c2.metric("Precision", f"{fm['precision']*100:.1f}%")
    c3.metric("Recall", f"{fm['recall']*100:.1f}%")
    c4.metric("F1-score", f"{fm['f1_score']*100:.1f}%")
    if "roc_auc" in fm:
        c5.metric("ROC-AUC", f"{fm['roc_auc']*100:.1f}%")

    st.subheader("Confusion Matrix (Test Set)")
    cm = np.array(meta["confusion_matrix"])
    cm_df = pd.DataFrame(cm, index=["Actual: FAIL", "Actual: PASS"],
                          columns=["Predicted: FAIL", "Predicted: PASS"])
    st.table(cm_df)

    st.subheader("Baseline vs. Tuned Model")
    bm = meta["baseline_metrics"]
    comp_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Untuned Baseline": [bm["accuracy"], bm["precision"], bm["recall"], bm["f1_score"]],
        "Tuned Final Model": [fm["accuracy"], fm["precision"], fm["recall"], fm["f1_score"]],
    })
    st.dataframe(comp_df, use_container_width=True)

    st.subheader("Early-Warning Comparison (without prior grades G1/G2)")
    ew = meta["early_warning_metrics"]
    st.write(
        "For comparison, a version of the model trained **without** G1/G2 "
        "(i.e. usable *before* any exam grade exists) achieves:"
    )
    c1, c2 = st.columns(2)
    c1.metric("Early-warning Accuracy", f"{ew['accuracy']*100:.1f}%",
              delta=f"{(ew['accuracy']-fm['accuracy'])*100:.1f} pts vs full model")
    c2.metric("Early-warning F1-score", f"{ew['f1_score']*100:.1f}%",
              delta=f"{(ew['f1_score']-fm['f1_score'])*100:.1f} pts vs full model")
    st.caption(
        "This illustrates a genuine trade-off: predicting earlier in the "
        "term (before formal grades exist) is possible but noticeably less "
        "accurate than waiting for the first two grading periods."
    )

    if meta.get("top_feature_importances"):
        st.subheader("Top Feature Importances")
        fi_df = pd.DataFrame(
            list(meta["top_feature_importances"].items()),
            columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=True)
        st.bar_chart(fi_df.set_index("Feature"))

    st.subheader("Known Model Limitations")
    st.markdown(
        """
        - Trained on **395 students from two Portuguese schools** (2008) —
          may not generalise to other regions, curricula or years.
        - Performance is strongly driven by **prior grades (G1, G2)**; the
          model is far less accurate when used purely as an early-term
          predictor (see comparison above).
        - The dataset is **moderately imbalanced** (67% pass / 33% fail).
        - Socio-demographic features (family status, parental job, etc.)
          risk encoding **historical bias**; they should never be used to
          justify treating a student differently on their own.
        """
    )


# --------------------------------------------------------------------------
# PAGE 4: RESPONSIBLE USE
# --------------------------------------------------------------------------
elif page == "⚠️ Responsible Use":
    st.title("⚠️ Responsible AI & Limitations")

    st.markdown(
        """
        This system is a **decision-support tool**, not a decision-making
        authority. It must never be presented to students, parents or staff
        as determining a student's future. Below are the key risks
        considered in its design, and how each is mitigated.
        """
    )

    risks = [
        ("Bias in student data",
         "The training data reflects the social and educational context of "
         "two specific schools in 2008. Patterns learned (e.g. around family "
         "background) may not be fair or accurate in other contexts.",
         "Predictions are framed as probabilities, not facts; the dashboard "
         "reports model limitations openly; retraining on local, "
         "up-to-date data is recommended before real deployment."),
        ("Privacy",
         "Student records include sensitive information (family status, "
         "alcohol use, health).",
         "The demo app processes data only in-memory for a single session; "
         "no student data is logged or transmitted externally; any real "
         "deployment must follow the school's data-protection policy and "
         "applicable law."),
        ("Data security",
         "A trained model and its inputs could be exposed if deployed "
         "without safeguards.",
         "Access to the app should be restricted to authorised staff; "
         "the saved model artefacts should not be shared outside the "
         "institution."),
        ("Incorrect predictions",
         "No model is 100% accurate (this model is ~90% accurate on held-out "
         "test data); false predictions in either direction are possible.",
         "Confidence scores are always shown alongside the prediction; the "
         "interface explicitly warns that the output is an estimate, not a "
         "certainty."),
        ("Over-reliance on AI",
         "Staff might be tempted to treat the prediction as authoritative "
         "and skip their own judgement.",
         "The Home and Predict pages carry an explicit disclaimer; the tool "
         "is positioned as one input among several, not a replacement for "
         "teacher assessment."),
        ("Fairness across groups",
         "Because features like sex, family structure and parental "
         "education are used, the model could perform differently across "
         "subgroups.",
         "Before any real deployment, performance should be checked "
         "separately across demographic subgroups (fairness auditing), and "
         "the model should be retrained/recalibrated if large gaps are "
         "found."),
    ]

    for title, risk, mitigation in risks:
        with st.expander(f"🔸 {title}"):
            st.markdown(f"**Risk:** {risk}")
            st.markdown(f"**Mitigation:** {mitigation}")

    st.markdown("---")
    st.error(
        "**This system must NOT be used as the sole basis for decisions "
        "about a student's grading, placement, discipline, or future "
        "opportunities.** It is a decision-support tool intended to "
        "complement, not replace, professional educational judgement."
    )
