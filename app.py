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

from predict import (predict_single, get_metadata, get_category_values,
                      predict_batch, get_feature_importance,
                      get_intervention_recommendations)  # noqa: E402
from config import (MODEL_PATH, REQUIRED_RAW_FIELDS, RP_INSTITUTIONS,
                    LEGACY_MODEL_COMPATIBILITY, RP_MODEL_AVAILABLE)  # noqa: E402
from services.student_service import StudentService  # noqa: E402

st.set_page_config(
    page_title="RP Student Success & Early Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="auto",
)


def inject_responsive_css():
    """
    Makes the app usable on phones and tablets, not just wide desktop
    screens. Streamlit's own column/radio layout is desktop-first by
    default (columns keep shrinking side-by-side instead of stacking, and
    horizontal radio groups can overflow), so this CSS adds explicit
    breakpoints:

      - > 768px (desktop/laptop): Streamlit's normal multi-column layout.
      - 481-768px (tablet / large phone): columns wrap to 2 per row.
      - <= 480px (phone): every column stacks to a single full-width column.

    Applied once, globally, so every page (Home, Predict, Dashboard,
    Responsible Use) benefits without page-specific changes.
    """
    st.markdown(
        """
        <style>
        /* Prevent mobile browsers from auto-zooming text and keep the
           page from ever scrolling sideways. */
        html, body { -webkit-text-size-adjust: 100%; }
        .main .block-container { overflow-x: hidden; }

        /* Fluid, readable headings instead of one fixed desktop size. */
        h1, [data-testid="stMarkdownContainer"] h1 { font-size: clamp(1.4rem, 4vw + 0.5rem, 2.25rem) !important; }
        h2, [data-testid="stMarkdownContainer"] h2 { font-size: clamp(1.15rem, 3vw + 0.4rem, 1.6rem) !important; }
        h3, [data-testid="stMarkdownContainer"] h3 { font-size: clamp(1.0rem, 2.2vw + 0.4rem, 1.3rem) !important; }

        /* Any row of st.columns() must wrap instead of squeezing forever. */
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            row-gap: 0.9rem;
        }

        /* Code / JSON blocks scroll horizontally instead of overflowing. */
        pre, code { overflow-x: auto !important; }

        /* Tablet & large phone: 2 columns per row, whatever N was. */
        @media (max-width: 768px) {
            .main .block-container {
                padding-left: 1.1rem !important;
                padding-right: 1.1rem !important;
                padding-top: 1.2rem !important;
            }
            div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                min-width: 46% !important;
                flex: 1 1 46% !important;
            }
            /* Horizontal radio groups wrap onto multiple lines instead of
               overflowing or forcing a sideways scroll. */
            div[role="radiogroup"] {
                flex-wrap: wrap !important;
                row-gap: 0.4rem !important;
            }
        }

        /* Phone: every column becomes full-width and stacks vertically. */
        @media (max-width: 480px) {
            .main .block-container {
                padding-left: 0.8rem !important;
                padding-right: 0.8rem !important;
            }
            div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                min-width: 100% !important;
                flex: 1 1 100% !important;
            }
            /* Metric values are large by default; shrink them so a 3-4
               metric row stays legible once stacked full-width. */
            div[data-testid="stMetricValue"] { font-size: 1.4rem !important; }
            div[data-testid="stMetricLabel"] { font-size: 0.8rem !important; }
            button[kind] { width: 100% !important; }
        }

        /* Dataframes/tables: scroll horizontally within their own box
           rather than forcing the whole page to scroll sideways. */
        div[data-testid="stDataFrame"], div[data-testid="stTable"] {
            overflow-x: auto !important;
            max-width: 100% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_responsive_css()


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
st.sidebar.title("🎓 Navigation")

# Handle navigation flags from session state (for auto-navigation from Student Portal)
if st.session_state.get("navigate_to_predict"):
    page = "🔮 Predict Performance"
    st.session_state["navigate_to_predict"] = False
elif st.session_state.get("navigate_to_whatif"):
    page = "🔬 What-If Analysis"
    st.session_state["navigate_to_whatif"] = False
else:
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "👨‍🎓 Student Portal", "📊 Dashboard", "🔮 Predict Performance",
         "⚠️ Early Warning", "👤 Student Profile", "🔬 What-If Analysis",
         "📋 Batch Prediction", "📈 Model Dashboard", "⚖️ Fairness Audit", "⚠️ Responsible Use"],
    )

if not MODEL_PATH.exists():
    st.sidebar.error(
        "No trained model found.\n\nRun `python src/train_model.py` first, "
        "then restart this app."
    )

st.sidebar.markdown("---")
st.sidebar.caption(
    "RP Student Success & Early Warning System · CAT2 Practical Project\n\n"
    "Module: Python and Fundamentals of AI (ITLPA701)"
)


# --------------------------------------------------------------------------
# PAGE 1: HOME
# --------------------------------------------------------------------------
if page == "🏠 Home":
    st.title("🎓 RP Student Success & Early Warning System")
    st.markdown("#### A Machine-Learning Decision-Support Tool for Academic Early Warning")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Project Description")
        st.write(
            "This system predicts whether a student is likely to **PASS** or **FAIL** "
            "their academic performance, based on academic, demographic, social "
            "and behavioural information. The prototype uses a legacy model trained on "
            "the **UCI Student Performance dataset** (Cortez & Silva, 2008) covering "
            "395 students from two Portuguese secondary schools. For Rwanda Polytechnic "
            "deployment, an RP-trained model is required."
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
        "Legacy model dataset citation: P. Cortez and A. Silva. *Using Data Mining to "
        "Predict Secondary School Student Performance.* In A. Brito and J. "
        "Teixeira (Eds.), Proceedings of 5th FUBUTEC 2008, pp. 5-12, Porto, "
        "Portugal, EUROSIS, 2008. "
        "Source: https://archive.ics.uci.edu/dataset/320/student+performance"
    )


# --------------------------------------------------------------------------
# PAGE 2: STUDENT PORTAL
# --------------------------------------------------------------------------
elif page == "👨‍🎓 Student Portal":
    st.title("👨‍🎓 RP Student Portal")
    st.markdown("#### Student Performance Prediction System")

    st.info(
        "**DEMO MODE**: This system uses demo student data for demonstration. "
        "Not connected to real RP production systems."
    )

    st.markdown("---")

    # Check if student is in session state
    if "selected_student" in st.session_state and st.session_state["selected_student"]:
        student = st.session_state["selected_student"]

        st.subheader("My Student Dashboard")

        # Student Information
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Student Information")
            st.write(f"**Name:** {student.get('full_name', 'N/A')}")
            st.write(f"**Registration Number:** {student.get('registration_number', 'N/A')}")
            st.write(f"**Institution / Campus:** {student.get('institution_name', 'N/A')} ({student.get('campus', 'N/A')})")
            st.write(f"**Programme:** {student.get('programme', 'N/A')}")
            st.write(f"**Department:** {student.get('department', 'N/A')}")

        with col2:
            st.markdown("### Academic Details")
            st.write(f"**Academic Year:** {student.get('academic_year', 'N/A')}")
            st.write(f"**Year of Study:** {student.get('year_of_study', 'N/A')}")
            st.write(f"**Semester:** {student.get('semester', 'N/A')}")

        st.markdown("---")

        # Performance Overview
        st.subheader("Performance Overview")

        # Check if academic data exists
        if "G1" in student and "G2" in student:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Period 1 Grade", f"{student['G1']}/20")
            with col2:
                st.metric("Period 2 Grade", f"{student['G2']}/20")
            with col3:
                avg_grade = (student['G1'] + student['G2']) / 2
                st.metric("Average Grade", f"{avg_grade:.1f}/20")

            if "absences" in student:
                st.write(f"**Absences:** {student['absences']}")
            if "failures" in student:
                st.write(f"**Past Failures:** {student['failures']}")
        else:
            st.info("Academic performance data not available in student record.")

        st.markdown("---")

        # Quick Actions
        st.subheader("Quick Actions")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Predict My Performance", use_container_width=True):
                st.session_state["navigate_to_predict"] = True
                st.rerun()
        with col2:
            if st.button("What-If Analysis", use_container_width=True):
                st.session_state["navigate_to_whatif"] = True
                st.rerun()
        with col3:
            if st.button("Clear Selection", use_container_width=True):
                del st.session_state["selected_student"]
                st.rerun()

        st.markdown("---")
        st.caption("Use the navigation sidebar to access other features.")

    else:
        # Empty state - no student selected
        st.subheader("Welcome to the RP Student Success Portal")

        st.write(
            "Use your Registration Number to access your student performance "
            "information and prediction."
        )

        st.markdown("---")

        st.subheader("Prediction Method")
        prediction_method = st.radio(
            "Select how you want to predict:",
            ["Use Registration Number", "Enter Information Manually"],
            horizontal=True
        )

        if prediction_method == "Use Registration Number":
            st.markdown("### Predict Using Registration Number")

            # Use RP_INSTITUTIONS from config as single source of truth
            institution_options = {
                inst["id"]: f"{inst['name']}"
                for inst in RP_INSTITUTIONS
            }

            col1, col2 = st.columns([1, 2])
            with col1:
                selected_institution_id = st.selectbox(
                    "Select RP Institution / College",
                    options=list(institution_options.keys()),
                    format_func=lambda x: institution_options[x]
                )
            with col2:
                registration_number = st.text_input(
                    "Registration Number",
                    placeholder="e.g., RP2024/1234",
                    max_chars=50
                )

            if st.button("Find Student", use_container_width=True):
                if not registration_number:
                    st.error("Please enter a Registration Number.")
                else:
                    try:
                        # Look up student
                        student = StudentService.find_student_by_registration(
                            registration_number, selected_institution_id
                        )

                        # Save to session state
                        st.session_state["selected_student"] = student
                        st.success("Student Found")
                        st.rerun()

                    except ValueError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

        else:
            # Manual prediction - redirect to existing Predict Performance page
            st.markdown("### Enter Information Manually")
            st.info("Please use the 'Predict Performance' page in the navigation for manual prediction.")
            st.markdown("Navigate to **🔮 Predict Performance** in the sidebar to enter student information manually.")


# --------------------------------------------------------------------------
# PAGE 3: DASHBOARD
# --------------------------------------------------------------------------
elif page == "📊 Dashboard":
    st.title("📊 Professional Dashboard")
    
    if not MODEL_PATH.exists():
        st.error("Model not found. Please run `python src/train_model.py` first.")
        st.stop()
    
    meta = get_metadata()
    
    # Load the full dataset to calculate KPIs
    from data_preprocessing import load_data, clean_data
    from feature_engineering import engineer_features, create_target
    
    raw_df = load_data()
    clean_df = clean_data(raw_df)
    clean_df = engineer_features(clean_df)
    clean_df = create_target(clean_df)
    
    # Calculate KPIs
    total_students = len(clean_df)
    pass_count = clean_df["pass_fail"].sum()
    fail_count = total_students - pass_count
    pass_rate = (pass_count / total_students) * 100
    fail_rate = (fail_count / total_students) * 100
    avg_grade = clean_df["G3"].mean()
    
    # High-risk students (predicted fail with high confidence)
    # We'll use absences > 10 and failures > 0 as a proxy for high risk
    high_risk_count = ((clean_df["absences"] > 10) | (clean_df["failures"] > 0)).sum()
    
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Students", f"{total_students}")
    with col2:
        st.metric("Pass Rate", f"{pass_rate:.1f}%")
    with col3:
        st.metric("Fail Rate", f"{fail_rate:.1f}%")
    with col4:
        st.metric("Average Final Grade", f"{avg_grade:.1f}/20")
    with col5:
        st.metric("High-Risk Students", f"{high_risk_count}")
    
    st.markdown("---")
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Pass/Fail Distribution")
        pass_fail_counts = clean_df["pass_fail"].value_counts()
        pass_fail_counts.index = ["PASS", "FAIL"]
        st.bar_chart(pass_fail_counts)
    
    with col2:
        st.subheader("Grade Distribution")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.hist(clean_df["G3"], bins=20, color="skyblue", edgecolor="black")
        ax.set_xlabel("Final Grade (G3)")
        ax.set_ylabel("Number of Students")
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Additional insights
    st.subheader("Student Demographics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**By School**")
        school_counts = clean_df["school"].value_counts()
        st.dataframe(school_counts)
    
    with col2:
        st.write("**By Sex**")
        sex_counts = clean_df["sex"].value_counts()
        st.dataframe(sex_counts)
    
    with col3:
        st.write("**By Address Type**")
        address_counts = clean_df["address"].value_counts()
        address_counts.index = ["Urban", "Rural"]
        st.dataframe(address_counts)
    
    st.markdown("---")
    st.info("💡 **Tip**: Use the navigation sidebar to explore other features like Early Warning, Student Profile, and What-If Analysis.")


# --------------------------------------------------------------------------
# PAGE 4: PREDICT
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
        # Note: 'school' field is for legacy model compatibility only
        # RP institutions are selected separately via RP_INSTITUTIONS
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

    # Model status warning
    if not RP_MODEL_AVAILABLE:
        st.warning(
            "⚠️ **Note**: This prototype uses a legacy model trained on Portuguese "
            "secondary-school data for demonstration purposes. "
            "An RP-trained model is not yet available. "
            "Predictions should be considered illustrative only."
        )

    # RP Institution mapping for demo purposes
    # Note: The legacy model was trained on Portuguese schools (GP/MS).
    # For RP demo, we map RP institutions to the model's expected values
    # using the centralized LEGACY_MODEL_COMPATIBILITY configuration.
    rp_institution_options = {
        inst["id"]: f"{inst['name']}"
        for inst in RP_INSTITUTIONS
    }

    with st.form("prediction_form"):
        st.markdown("##### 👤 Demographics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            selected_rp_institution = st.selectbox(
                "RP Institution / College",
                options=list(rp_institution_options.keys()),
                format_func=lambda x: rp_institution_options[x]
            )
            # Map to legacy model's school field using compatibility layer
            school = LEGACY_MODEL_COMPATIBILITY.get(selected_rp_institution, "GP")
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
        # Note: 'school' is already the legacy model code (GP/MS) from LEGACY_MODEL_COMPATIBILITY
        raw_input = dict(
            school=school,  # Already mapped via LEGACY_MODEL_COMPATIBILITY
            sex=LABELS["sex"][sex], age=age,
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

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if result["prediction"] == "PASS":
                st.success(f"### Prediction: {result['prediction']} ✅")
            else:
                st.error(f"### Prediction: {result['prediction']} ⚠️")
        with col2:
            st.metric("Confidence", f"{result['confidence']}%")
        with col3:
            st.metric("P(Pass)", f"{result['probability_pass']}%")
        with col4:
            risk_color = "🟢" if result["risk_level"] == "Low" else "🟡" if result["risk_level"] == "Medium" else "🔴"
            st.metric("Risk Level", f"{risk_color} {result['risk_level']}")

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

        # Feature importance explanation
        st.markdown("---")
        st.subheader("Key Factors Influencing Prediction")
        feature_importance = get_feature_importance()
        if feature_importance:
            # Show top 5 features
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for feature, importance in top_features:
                st.write(f"• **{feature}**: {importance:.4f}")
        
        # Intervention recommendations
        st.markdown("---")
        st.subheader("Recommended Interventions")
        recommendations = get_intervention_recommendations(raw_input, result)
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")

        st.warning(
            "⚠️ This is a probabilistic estimate from a statistical model, "
            "**not a certainty and not a decision**. Please combine this "
            "result with your own professional judgement."
        )


# --------------------------------------------------------------------------
# PAGE 4: EARLY WARNING
# --------------------------------------------------------------------------
elif page == "⚠️ Early Warning":
    st.title("⚠️ Early Warning System")
    
    if not MODEL_PATH.exists():
        st.error("Model not found. Please run `python src/train_model.py` first.")
        st.stop()
    
    st.write("Identify students at risk of failing based on key risk factors.")
    
    from data_preprocessing import load_data, clean_data
    from feature_engineering import engineer_features, create_target
    
    raw_df = load_data()
    clean_df = clean_data(raw_df)
    clean_df = engineer_features(clean_df)
    clean_df = create_target(clean_df)
    
    # Define risk factors
    st.subheader("Risk Factors")
    col1, col2 = st.columns(2)
    
    with col1:
        high_absences = st.slider("High Absences Threshold", 5, 20, 10)
        past_failures = st.slider("Past Failures Threshold", 0, 3, 1)
    
    with col2:
        low_study_time = st.slider("Low Study Time (hours)", 1, 4, 2)
        low_grades = st.slider("Low Prior Grade Threshold", 0, 10, 8)
    
    # Identify at-risk students
    at_risk = clean_df[
        (clean_df["absences"] >= high_absences) |
        (clean_df["failures"] >= past_failures) |
        (clean_df["studytime"] <= low_study_time) |
        (clean_df["G1"] <= low_grades) |
        (clean_df["G2"] <= low_grades)
    ].copy()
    
    # Calculate risk score for each student
    at_risk["risk_score"] = (
        (at_risk["absences"] >= high_absences).astype(int) * 2 +
        (at_risk["failures"] >= past_failures).astype(int) * 3 +
        (at_risk["studytime"] <= low_study_time).astype(int) * 1 +
        (at_risk["G1"] <= low_grades).astype(int) * 2 +
        (at_risk["G2"] <= low_grades).astype(int) * 2
    )
    
    st.markdown("---")
    st.subheader(f"At-Risk Students ({len(at_risk)} identified)")
    
    if len(at_risk) > 0:
        # Sort by risk score
        at_risk = at_risk.sort_values("risk_score", ascending=False)
        
        # Display top 20 at-risk students
        display_cols = ["school", "sex", "age", "absences", "failures", 
                        "studytime", "G1", "G2", "G3", "risk_score"]
        st.dataframe(at_risk[display_cols].head(20), use_container_width=True)
        
        st.markdown("---")
        st.subheader("Intervention Recommendations")
        
        # Show sample interventions for top 5 at-risk students
        for idx, row in at_risk.head(5).iterrows():
            st.write(f"**Student ID {idx}** (Risk Score: {row['risk_score']})")
            student_data = row.to_dict()
            # Create a mock prediction result for recommendations
            mock_result = {
                "risk_level": "High" if row["risk_score"] >= 5 else "Medium"
            }
            recommendations = get_intervention_recommendations(student_data, mock_result)
            for rec in recommendations[:3]:
                st.write(f"  • {rec}")
            st.write("")
    else:
        st.success("No students match the current risk criteria. Adjust the thresholds to identify more students.")


# --------------------------------------------------------------------------
# PAGE 6: STUDENT PROFILE
# --------------------------------------------------------------------------
elif page == "👤 Student Profile":
    st.title("👤 Student Profile")
    
    if not MODEL_PATH.exists():
        st.error("Model not found. Please run `python src/train_model.py` first.")
        st.stop()
    
    from data_preprocessing import load_data, clean_data
    from feature_engineering import engineer_features, create_target
    
    raw_df = load_data()
    clean_df = clean_data(raw_df)
    clean_df = engineer_features(clean_df)
    clean_df = create_target(clean_df)
    
    st.write("Select a student to view their detailed profile and prediction.")
    
    # Student selector
    student_ids = clean_df.index.tolist()
    selected_idx = st.selectbox("Select Student by Index", student_ids)
    
    student = clean_df.loc[selected_idx]
    
    st.markdown("---")
    
    # Display student information in organized sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Academic Information")
        st.write(f"**School:** {student['school']}")
        st.write(f"**Period 1 Grade (G1):** {student['G1']}/20")
        st.write(f"**Period 2 Grade (G2):** {student['G2']}/20")
        st.write(f"**Final Grade (G3):** {student['G3']}/20")
        st.write(f"**Average Prior Grade:** {student['average_prior_grade']:.1f}/20")
        st.write(f"**Grade Trend:** {'Improving' if student['grade_trend'] > 0 else 'Declining' if student['grade_trend'] < 0 else 'Stable'}")
        st.write(f"**Study Time:** {student['studytime']} hours/week")
        st.write(f"**Absences:** {student['absences']}")
        st.write(f"**Past Failures:** {student['failures']}")
    
    with col2:
        st.subheader("Demographics & Family")
        st.write(f"**Age:** {student['age']}")
        st.write(f"**Sex:** {student['sex']}")
        st.write(f"**Address:** {'Urban' if student['address'] == 'U' else 'Rural'}")
        st.write(f"**Family Size:** {'>3 members' if student['famsize'] == 'GT3' else '≤3 members'}")
        st.write(f"**Parents' Status:** {'Living together' if student['Pstatus'] == 'T' else 'Living apart'}")
        st.write(f"**Mother's Education:** {student['Medu']}/4")
        st.write(f"**Father's Education:** {student['Fedu']}/4")
        st.write(f"**Mother's Job:** {student['Mjob']}")
        st.write(f"**Father's Job:** {student['Fjob']}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Support & Activities")
        st.write(f"**School Support:** {'Yes' if student['schoolsup'] == 'yes' else 'No'}")
        st.write(f"**Family Support:** {'Yes' if student['famsup'] == 'yes' else 'No'}")
        st.write(f"**Paid Classes:** {'Yes' if student['paid'] == 'yes' else 'No'}")
        st.write(f"**Extracurricular Activities:** {'Yes' if student['activities'] == 'yes' else 'No'}")
        st.write(f"**Wants Higher Education:** {'Yes' if student['higher'] == 'yes' else 'No'}")
        st.write(f"**Internet Access:** {'Yes' if student['internet'] == 'yes' else 'No'}")
    
    with col2:
        st.subheader("Social & Wellbeing")
        st.write(f"**Family Relationship Quality:** {student['famrel']}/5")
        st.write(f"**Free Time:** {student['freetime']}/5")
        st.write(f"**Going Out with Friends:** {student['goout']}/5")
        st.write(f"**Workday Alcohol:** {student['Dalc']}/5")
        st.write(f"**Weekend Alcohol:** {student['Walc']}/5")
        st.write(f"**Health Status:** {student['health']}/5")
    
    st.markdown("---")
    
    # Make prediction for this student
    st.subheader("Prediction & Risk Assessment")
    
    # Convert student data to raw input format
    raw_input = {
        "school": student["school"], "sex": student["sex"], "age": int(student["age"]),
        "address": student["address"], "famsize": student["famsize"], "Pstatus": student["Pstatus"],
        "Medu": int(student["Medu"]), "Fedu": int(student["Fedu"]), "Mjob": student["Mjob"],
        "Fjob": student["Fjob"], "reason": student["reason"], "guardian": student["guardian"],
        "traveltime": int(student["traveltime"]), "studytime": int(student["studytime"]),
        "failures": int(student["failures"]), "schoolsup": student["schoolsup"],
        "famsup": student["famsup"], "paid": student["paid"], "activities": student["activities"],
        "nursery": student["nursery"], "higher": student["higher"], "internet": student["internet"],
        "romantic": student["romantic"], "famrel": int(student["famrel"]),
        "freetime": int(student["freetime"]), "goout": int(student["goout"]),
        "Dalc": int(student["Dalc"]), "Walc": int(student["Walc"]), "health": int(student["health"]),
        "absences": int(student["absences"]), "G1": int(student["G1"]), "G2": int(student["G2"]),
    }
    
    try:
        result = predict_single(raw_input)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if result["prediction"] == "PASS":
                st.success(f"### Prediction: {result['prediction']} ✅")
            else:
                st.error(f"### Prediction: {result['prediction']} ⚠️")
        with col2:
            st.metric("Confidence", f"{result['confidence']}%")
        with col3:
            risk_color = "🟢" if result["risk_level"] == "Low" else "🟡" if result["risk_level"] == "Medium" else "🔴"
            st.metric("Risk Level", f"{risk_color} {result['risk_level']}")
        
        # Show recommendations
        st.markdown("---")
        st.subheader("Recommended Interventions")
        recommendations = get_intervention_recommendations(raw_input, result)
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")


# --------------------------------------------------------------------------
# PAGE 6: WHAT-IF ANALYSIS
# --------------------------------------------------------------------------
elif page == "🔬 What-If Analysis":
    st.title("🔬 What-If Analysis")
    
    if not MODEL_PATH.exists():
        st.error("Model not found. Please run `python src/train_model.py` first.")
        st.stop()
    
    st.write("Explore how changing student factors affects the predicted outcome.")
    
    # Load a sample student as baseline
    from data_preprocessing import load_data, clean_data
    raw_df = load_data()
    clean_df = clean_data(raw_df)
    
    # Use the first student as baseline
    baseline = clean_df.iloc[0].to_dict()
    
    st.markdown("---")
    st.subheader("Adjust Student Factors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Academic Factors**")
        new_studytime = st.slider("Study Time (hours/week)", 1, 4, int(baseline["studytime"]))
        new_failures = st.slider("Past Failures", 0, 3, int(baseline["failures"]))
        new_absences = st.slider("Absences", 0, 75, int(baseline["absences"]))
        new_G1 = st.slider("Period 1 Grade (G1)", 0, 20, int(baseline["G1"]))
        new_G2 = st.slider("Period 2 Grade (G2)", 0, 20, int(baseline["G2"]))
    
    with col2:
        st.write("**Support Factors**")
        new_schoolsup = st.radio("School Support", ["yes", "no"], 
                                  index=0 if baseline["schoolsup"] == "yes" else 1)
        new_famsup = st.radio("Family Support", ["yes", "no"],
                               index=0 if baseline["famsup"] == "yes" else 1)
        new_paid = st.radio("Paid Classes", ["yes", "no"],
                            index=0 if baseline["paid"] == "yes" else 1)
    
    # Create modified input
    modified_input = baseline.copy()
    modified_input.update({
        "studytime": new_studytime,
        "failures": new_failures,
        "absences": new_absences,
        "G1": new_G1,
        "G2": new_G2,
        "schoolsup": new_schoolsup,
        "famsup": new_famsup,
        "paid": new_paid,
    })
    
    # Get predictions for baseline and modified
    try:
        baseline_result = predict_single(baseline)
        modified_result = predict_single(modified_input)
        
        st.markdown("---")
        st.subheader("Prediction Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Baseline (Original)**")
            st.metric("Prediction", baseline_result["prediction"])
            st.metric("P(Pass)", f"{baseline_result['probability_pass']}%")
            st.metric("Risk Level", baseline_result["risk_level"])
        
        with col2:
            st.write("**Modified (What-If)**")
            delta_color = "normal" if modified_result["probability_pass"] >= baseline_result["probability_pass"] else "inverse"
            st.metric("Prediction", modified_result["prediction"])
            st.metric("P(Pass)", f"{modified_result['probability_pass']}%",
                     delta=f"{modified_result['probability_pass'] - baseline_result['probability_pass']:.1f}%",
                     delta_color=delta_color)
            st.metric("Risk Level", modified_result["risk_level"])
        
        st.markdown("---")
        st.subheader("Analysis")
        if modified_result["probability_pass"] > baseline_result["probability_pass"]:
            st.success(f"Improving these factors increases the probability of passing by {modified_result['probability_pass'] - baseline_result['probability_pass']:.1f} percentage points.")
        elif modified_result["probability_pass"] < baseline_result["probability_pass"]:
            st.warning(f"These changes decrease the probability of passing by {baseline_result['probability_pass'] - modified_result['probability_pass']:.1f} percentage points.")
        else:
            st.info("These changes do not significantly affect the prediction.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")


# --------------------------------------------------------------------------
# PAGE 7: BATCH PREDICTION
# --------------------------------------------------------------------------
elif page == "📋 Batch Prediction":
    st.title("📋 Batch Prediction")
    
    if not MODEL_PATH.exists():
        st.error("Model not found. Please run `python src/train_model.py` first.")
        st.stop()
    
    st.write("Upload a CSV file with multiple student records to get predictions for all of them.")
    
    st.markdown("---")
    st.subheader("Upload CSV File")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df)} student records.")
            st.dataframe(df.head(), use_container_width=True)
            
            # Validate required columns
            required_cols = set(REQUIRED_RAW_FIELDS)
            missing_cols = required_cols - set(df.columns)
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                st.stop()
            
            if st.button("🔮 Predict All Records"):
                with st.spinner("Processing predictions..."):
                    # Convert DataFrame rows to list of dicts
                    records = df.to_dict("records")
                    results = predict_batch(records)
                    
                    # Add results to DataFrame
                    df["prediction"] = [r.get("prediction", "ERROR") for r in results]
                    df["confidence"] = [r.get("confidence", 0) for r in results]
                    df["probability_pass"] = [r.get("probability_pass", 0) for r in results]
                    df["probability_fail"] = [r.get("probability_fail", 0) for r in results]
                    df["risk_level"] = [r.get("risk_level", "Unknown") for r in results]
                    
                    st.markdown("---")
                    st.subheader("Prediction Results")
                    st.dataframe(df, use_container_width=True)
                    
                    # Summary statistics
                    st.markdown("---")
                    st.subheader("Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        pass_count = (df["prediction"] == "PASS").sum()
                        st.metric("Predicted PASS", pass_count)
                    with col2:
                        fail_count = (df["prediction"] == "FAIL").sum()
                        st.metric("Predicted FAIL", fail_count)
                    with col3:
                        high_risk = (df["risk_level"] == "High").sum()
                        st.metric("High Risk", high_risk)
                    with col4:
                        error_count = (df["prediction"] == "ERROR").sum()
                        st.metric("Errors", error_count)
                    
                    # Download options
                    st.markdown("---")
                    st.subheader("Download Results")
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="batch_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Show errors if any
                    if error_count > 0:
                        st.error(f"{error_count} records had errors. Check the 'prediction' column for details.")
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    st.markdown("---")
    st.info("💡 **Tip**: The CSV file must contain all required columns matching the training data format. Download the template below for reference.")
    
    # Provide template download
    from data_preprocessing import load_data
    template_df = load_data().head(1)
    template_csv = template_df.to_csv(index=False)
    st.download_button(
        label="Download Template CSV",
        data=template_csv,
        file_name="student_template.csv",
        mime="text/csv"
    )


# --------------------------------------------------------------------------
# PAGE 9: MODEL DASHBOARD (renumbered)
# --------------------------------------------------------------------------
elif page == "📈 Model Dashboard":
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
    # st.dataframe (not st.table) scrolls/resizes within its own box on a
    # narrow screen instead of a fixed-width table forcing the page to
    # scroll sideways.
    st.dataframe(cm_df, use_container_width=True)

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
# PAGE 9: FAIRNESS AUDIT
# --------------------------------------------------------------------------
elif page == "⚖️ Fairness Audit":
    st.title("⚖️ Fairness Audit")
    
    if not MODEL_PATH.exists():
        st.error("Model not found. Please run `python src/train_model.py` first.")
        st.stop()
    
    st.write("Performance analysis across demographic subgroups to identify potential bias.")
    
    # Load fairness audit report
    from config import FAIRNESS_REPORT_MD, FAIRNESS_REPORT_JSON
    
    if FAIRNESS_REPORT_MD.exists():
        st.markdown("---")
        st.subheader("Fairness Audit Report")
        
        with open(FAIRNESS_REPORT_MD) as f:
            report_content = f.read()
        st.markdown(report_content)
    else:
        st.warning("Fairness audit report not found. Run `python src/fairness_audit.py` to generate it.")
    
    if FAIRNESS_REPORT_JSON.exists():
        st.markdown("---")
        st.subheader("Detailed Metrics (JSON)")
        with open(FAIRNESS_REPORT_JSON) as f:
            fairness_data = json.load(f)
        st.json(fairness_data)
    
    st.markdown("---")
    st.info("💡 **Note**: This audit uses a single test split. For production deployment, repeat this audit with cross-validation on larger, more current data.")


# --------------------------------------------------------------------------
# PAGE 11: RESPONSIBLE USE
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
        ("Legacy model training data",
         "The current model was trained on Portuguese secondary school data (2008). "
         "It is NOT an RP-trained model. RP institution data is mapped for prototype "
         "compatibility via LEGACY_MODEL_COMPATIBILITY. An RP-trained model does not yet exist.",
         "This limitation is documented in MODEL_CARD.md and README.md. "
         "Production deployment for Rwanda Polytechnic requires validation and/or "
         "retraining using representative RP student data to ensure cultural and "
         "educational context alignment."),
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
