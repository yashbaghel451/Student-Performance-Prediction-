"""
Student Performance Prediction using Machine Learning
Note: The included dataset is synthetic/demo data for an academic mini-project.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

DATA_FILE = "student_performance.csv"

def main():
    try:
        data = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Dataset not found: {DATA_FILE}. Keep it in the same folder as main.py.")
        return

    features = ["study_hours_per_week", "attendance_percent", "previous_marks"]
    target = "final_marks"

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("\n=== Student Performance Prediction ===")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, predictions):.2f}")
    print(f"R-squared (R2): {r2_score(y_test, predictions):.3f}")
    print("\nEnter a student's details to estimate final marks.")
    print("This is a learning demo, not a reliable real-world assessment tool.")

    try:
        study_hours = float(input("Study hours per week (0-80): "))
        attendance = float(input("Attendance percentage (0-100): "))
        previous_marks = float(input("Previous marks (0-100): "))
    except ValueError:
        print("Please enter numeric values.")
        return

    if not (0 <= study_hours <= 80 and 0 <= attendance <= 100 and 0 <= previous_marks <= 100):
        print("Input out of range. Please use the ranges shown above.")
        return

    sample = pd.DataFrame([{
        "study_hours_per_week": study_hours,
        "attendance_percent": attendance,
        "previous_marks": previous_marks
    }])
    result = float(model.predict(sample)[0])
    result = max(0.0, min(100.0, result))
    print(f"\nEstimated final marks: {result:.1f}/100")
    print("Reminder: actual results depend on many factors; this estimate is not a guarantee.")

if __name__ == "__main__":
    main()
