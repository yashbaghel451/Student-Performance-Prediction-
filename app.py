
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# LOAD DATA AND TRAIN MODEL
# ==================================================

DATA_PATH = Path(__file__).parent / "student_performance.csv"

FEATURES = [
    "study_hours_per_week",
    "attendance_percent",
    "previous_marks",
]

TARGET = "final_marks"


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [TARGET]
    missing = [col for col in required_columns if col not in data.columns]

    if missing:
        raise ValueError(
            "Missing CSV columns: " + ", ".join(missing)
        )

    data = data[required_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).dropna()

    if data.empty:
        raise ValueError("No valid training records found.")

    return data


@st.cache_resource
def train_model():
    data = load_data()

    model = LinearRegression()
    model.fit(data[FEATURES], data[TARGET])

    return model


try:
    df = load_data()
    model = train_model()
except Exception as error:
    st.error(f"Unable to load project data: {error}")
    st.stop()


# ==================================================
# PROFESSIONAL WHITE AND BLUE THEME
# ==================================================

st.markdown(
    """
    <style>
    :root {
        --primary: #2563EB;
        --primary-dark: #1D4ED8;
        --text: #172B4D;
        --muted: #64748B;
        --border: #E2E8F0;
        --surface: #FFFFFF;
        --background: #F5F8FF;
    }

    .stApp {
        background: var(--background);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(245, 248, 255, 0.96);
    }

    .block-container {
        max-width: 1320px;
        padding: 1.7rem 2rem 3rem 2rem;
    }

    /* SIDEBAR */

    [data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #334155 !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #163A70 !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 0.55rem;
    }

    [data-testid="stSidebar"] [role="radio"] {
        padding: 0.55rem 0.65rem;
        border-radius: 8px;
    }

    [data-testid="stSidebar"] [role="radio"]:hover {
        background: #EFF6FF;
    }

    /* HERO HEADER */

    .hero {
        background: linear-gradient(120deg, #1D4ED8, #3B82F6);
        border-radius: 18px;
        padding: 30px 32px;
        margin-bottom: 26px;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.13);
    }

    .hero h1 {
        color: #FFFFFF !important;
        font-size: clamp(25px, 3vw, 34px);
        font-weight: 750;
        line-height: 1.25;
        margin: 0 0 10px 0;
    }

    .hero p {
        color: #EAF2FF !important;
        font-size: 15px;
        line-height: 1.6;
        margin: 0;
    }

    /* SECTION HEADINGS */

    .section-title {
        color: #172B4D;
        font-size: 23px;
        font-weight: 750;
        margin: 28px 0 5px 0;
    }

    .section-subtitle {
        color: #64748B;
        font-size: 14px;
        line-height: 1.7;
        margin: 0 0 20px 0;
    }

    h2, h3, h4 {
        color: #172B4D;
    }

    /* METRIC CARDS */

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2EAF5;
        border-radius: 14px;
        padding: 19px;
        min-height: 125px;
        box-shadow: 0 4px 14px rgba(30, 64, 120, 0.045);
    }

    [data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-size: 13px;
    }

    [data-testid="stMetricValue"] {
        color: #2563EB !important;
        font-weight: 700;
    }

    /* OVERVIEW CARDS */

    .info-card {
        background: #FFFFFF;
        border: 1px solid #E2EAF5;
        border-radius: 14px;
        padding: 22px;
        min-height: 142px;
        height: 100%;
        box-shadow: 0 4px 14px rgba(30, 64, 120, 0.035);
    }

    .info-card h3 {
        color: #2563EB !important;
        font-size: 17px;
        margin: 0 0 12px 0;
    }

    .info-card p {
        color: #64748B !important;
        font-size: 14px;
        line-height: 1.7;
        margin: 0;
    }

    /* BUTTONS */

    .stButton > button,
    [data-testid="stFormSubmitButton"] > button {
        background: #2563EB;
        color: #FFFFFF !important;
        border: 1px solid #2563EB;
        border-radius: 9px;
        min-height: 45px;
        font-weight: 650;
        transition: background 0.2s ease;
    }

    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        background: #1D4ED8;
        border-color: #1D4ED8;
        color: #FFFFFF !important;
    }

    /* INPUTS */

    [data-testid="stNumberInput"] input {
        background: #FFFFFF;
        color: #172B4D;
        border-radius: 8px;
    }

    [data-testid="stNumberInput"] label {
        color: #334155 !important;
    }

    /* PREDICTION RESULT */

    .result-card {
        background: #FFFFFF;
        border: 1px solid #D9E6FF;
        border-left: 5px solid #2563EB;
        border-radius: 14px;
        padding: 24px;
        margin: 22px 0 16px 0;
        box-shadow: 0 5px 18px rgba(37, 99, 235, 0.06);
    }

    .result-label {
        color: #64748B !important;
        font-size: 14px;
    }

    .result-score {
        color: #2563EB !important;
        font-size: 42px;
        font-weight: 800;
        line-height: 1.4;
        margin: 5px 0;
    }

    /* DATA TABLE */

    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        overflow: hidden;
    }

    /* PROGRESS BAR */

    [data-testid="stProgressBar"] > div > div {
        background: #2563EB;
    }

    /* MOBILE AND SMALL WINDOWS */

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 1rem 2rem 1rem;
        }

        .hero {
            padding: 23px 20px;
        }

        .info-card {
            min-height: auto;
        }

        .result-score {
            font-size: 34px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

with st.sidebar:
    st.markdown("## 🎓 Student AI")
    st.caption("Performance Analytics Platform")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Predict Student", "Dataset"],
        label_visibility="visible",
    )

    st.divider()

    st.markdown("### Machine Learning")
    st.caption("Algorithm: Linear Regression")
    st.caption("Dataset: student_performance.csv")

    st.divider()

    st.caption("Student Performance Prediction")
    st.caption("Academic ML Project")


# ==================================================
# REUSABLE HEADER
# ==================================================

def show_header(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================
# DASHBOARD PAGE
# ==================================================

if page == "Dashboard":

    show_header(
        "Student Performance Dashboard",
        "Explore student data and estimate academic performance.",
    )

    total_students = len(df)
    average_marks = df[TARGET].mean()
    highest_marks = df[TARGET].max()
    average_attendance = df["attendance_percent"].mean()

    c1, c2, c3, c4 = st.columns(4, gap="medium")

    c1.metric("Training Records", f"{total_students}")
    c2.metric("Average Final Marks", f"{average_marks:.1f}/100")
    c3.metric("Highest Final Marks", f"{highest_marks:.1f}/100")
    c4.metric("Average Attendance", f"{average_attendance:.1f}%")

    st.markdown(
        '<div class="section-title">Project Overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'A machine learning application that estimates final marks '
        'using student-related input features.'
        '</div>',
        unsafe_allow_html=True,
    )

    a, b, c = st.columns(3, gap="medium")

    with a:
        st.markdown(
            """
            <div class="info-card">
                <h3>📚 Study Data</h3>
                <p>Uses the study-hours feature recorded in the dataset.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b:
        st.markdown(
            """
            <div class="info-card">
                <h3>📅 Attendance</h3>
                <p>Includes attendance percentage as a model input.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c:
        st.markdown(
            """
            <div class="info-card">
                <h3>🤖 ML Prediction</h3>
                <p>Uses Linear Regression to estimate final marks.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">Final Marks Distribution</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Final marks recorded in the training dataset.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Explicit chart colors make it match the light dashboard.
    chart_data = df[[TARGET]].reset_index(drop=True)
    chart_data.index = chart_data.index + 1
    chart_data.index.name = "Training Record"

    st.bar_chart(
        chart_data,
        color="#3B82F6",
        use_container_width=True,
    )

    st.info(
        "Dataset statistics describe the available training records. "
        "Predictions are estimates and are not guaranteed student results."
    )


# ==================================================
# PREDICTION PAGE
# ==================================================

elif page == "Predict Student":

    show_header(
        "Student Performance Prediction",
        "Enter student details to generate a model-based estimate.",
    )

    st.markdown("### Student Information")

    st.markdown(
        '<div class="section-subtitle">'
        'Enter values within the ranges represented in the training dataset.'
        '</div>',
        unsafe_allow_html=True,
    )

    def feature_bounds(column):
        low = float(df[column].min())
        high = float(df[column].max())

        if low == high:
            high = low + 1.0

        return low, high

    study_min, study_max = feature_bounds("study_hours_per_week")
    attendance_min, attendance_max = feature_bounds("attendance_percent")
    marks_min, marks_max = feature_bounds("previous_marks")

    with st.form("prediction_form"):

        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("#### 📚 Study Details")

            study_hours = st.number_input(
                "Study hours (dataset units)",
                min_value=study_min,
                max_value=study_max,
                value=float(df["study_hours_per_week"].median()),
                step=1.0,
                help="Range is based on the existing training data.",
            )

            attendance = st.number_input(
                "Attendance (%)",
                min_value=attendance_min,
                max_value=attendance_max,
                value=float(df["attendance_percent"].median()),
                step=1.0,
            )

        with right:
            st.markdown("#### 📝 Academic Details")

            previous_marks = st.number_input(
                "Previous marks",
                min_value=marks_min,
                max_value=marks_max,
                value=float(df["previous_marks"].median()),
                step=1.0,
            )

            st.markdown(
                """
                <div class="info-card">
                    <h3>Prediction Model</h3>
                    <p><b>Algorithm:</b> Linear Regression</p>
                    <p><b>Output:</b> Estimated final marks</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        submitted = st.form_submit_button(
            "✨ Predict Performance",
            use_container_width=True,
        )

    if submitted:

        input_data = pd.DataFrame(
            [[study_hours, attendance, previous_marks]],
            columns=FEATURES,
        )

        prediction = float(model.predict(input_data)[0])

        # Keep the displayed marks on the 0-100 scale.
        displayed_prediction = max(0.0, min(100.0, prediction))

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Estimated Final Marks</div>
                <div class="result-score">
                    {displayed_prediction:.1f}
                    <span style="font-size:20px">/ 100</span>
                </div>
                <div class="result-label">
                    Generated by your Linear Regression model
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(displayed_prediction / 100.0)

        st.markdown("### Input Summary")

        r1, r2, r3 = st.columns(3, gap="medium")

        r1.metric("Study Hours", f"{study_hours:g}")
        r2.metric("Attendance", f"{attendance:g}%")
        r3.metric("Previous Marks", f"{previous_marks:g}")

        st.caption(
            "This prediction is an estimate based on the available data. "
            "It is not a guaranteed result or a measure of student ability."
        )


# ==================================================
# DATASET PAGE
# ==================================================

elif page == "Dataset":

    show_header(
        "Training Dataset",
        "Explore the records used to train the prediction model.",
    )

    d1, d2 = st.columns(2, gap="medium")

    d1.metric("Valid Training Rows", len(df))
    d2.metric("Model Input Features", len(FEATURES))

    st.markdown("### Dataset Preview")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Feature Summary")

    st.dataframe(
        df.describe().round(2),
        use_container_width=True,
    )

    st.warning(
        "Before submission, verify that study_hours_per_week contains "
        "realistic values and that its units are correct. This application "
        "uses the original dataset values without automatically changing them."
    )
