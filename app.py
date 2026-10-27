
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Prediction")
st.write(
    "Predict estimated final marks using study hours, "
    "attendance, and previous marks."
)

@st.cache_data
def load_data():
    return pd.read_csv("student_performance.csv")

df = load_data()

features = [
    "study_hours_per_week",
    "attendance_percent",
    "previous_marks"
]

X = df[features]
y = df["final_marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

st.subheader("Model Performance")

col1, col2 = st.columns(2)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("R² Score", f"{r2:.3f}")

st.divider()
st.subheader("Enter Student Details")

with st.form("prediction_form"):
    study_hours = st.number_input(
        "Study hours per week",
        min_value=0,
        max_value=100,
        value=60
    )

    attendance = st.number_input(
        "Attendance percentage",
        min_value=0,
        max_value=100,
        value=65
    )

    previous_marks = st.number_input(
        "Previous marks",
        min_value=0,
        max_value=100,
        value=52
    )

    submitted = st.form_submit_button("Predict Final Marks")

if submitted:
    input_data = pd.DataFrame(
        [[study_hours, attendance, previous_marks]],
        columns=features
    )

    result = model.predict(input_data)[0]
    result = max(0, min(100, result))

    st.success(f"Estimated Final Marks: {result:.1f}/100")

    st.caption(
        "This is an educational ML demo. "
        "Predictions are not guaranteed real-world results."
    )

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)
