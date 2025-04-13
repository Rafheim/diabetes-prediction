import streamlit as st
import pandas as pd
import joblib
import traceback

# Load model and preprocessor
model = joblib.load("diabetes_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# App Title
st.title("🩺 Diabetes Prediction App")
st.write("Enter patient data below to check the likelihood of diabetes.")

# User Input Form
with st.form("prediction_form"):
    year = st.number_input("Year", min_value=2000, max_value=2030, value=2023)
    age = st.number_input("Age", min_value=1, value=45)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    race = st.selectbox("Race", ["AfricanAmerican", "Asian", "Caucasian", "Hispanic", "Other"])
    hypertension = st.radio("Hypertension", ["No", "Yes"])
    heart_disease = st.radio("Heart Disease", ["No", "Yes"])
    bmi = st.number_input("BMI", min_value=1.0, value=28.0)
    hbA1c_level = st.number_input("HbA1c Level", min_value=0.0, value=5.8)
    blood_glucose_level = st.number_input("Blood Glucose Level", min_value=0, value=100)
    location = st.selectbox("Location", [
        "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
        "Connecticut", "Delaware", "District of Columbia", "Florida", "Georgia", "Guam",
        "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
        "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
        "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
        "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
        "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island",
        "South Carolina", "South Dakota", "Tennessee", "Texas", "United States", "Utah",
        "Vermont", "Virgin Islands", "Virginia", "Washington", "West Virginia",
        "Wisconsin", "Wyoming"
    ])
    smoking_history = st.selectbox("Smoking History", ["never", "current", "former", "not current", "ever", "No Info"])

    submit = st.form_submit_button("Predict")

if submit:
    # Convert 'Yes'/'No' to binary
    hypertension_bin = 1 if hypertension == "Yes" else 0
    heart_disease_bin = 1 if heart_disease == "Yes" else 0

    # Input dictionary
    input_dict = {
        "year": year,
        "age": age,
        "hypertension": hypertension_bin,
        "heart_disease": heart_disease_bin,
        "bmi": bmi,
        "hbA1c_level": hbA1c_level,
        "blood_glucose_level": blood_glucose_level,
        "gender": gender,
        "location": location,
        "smoking_history": smoking_history,
    }

    # Manually one-hot encode race
    race_options = ["AfricanAmerican", "Asian", "Caucasian", "Hispanic", "Other"]
    for r in race_options:
        input_dict[f"race:{r}"] = 1 if race == r else 0

    # Create DataFrame
    raw_input = pd.DataFrame([input_dict])

    try:
        # Transform using preprocessor
        processed_input = preprocessor.transform(raw_input)

        # Convert to DataFrame and force all expected columns
        expected_cols = preprocessor.get_feature_names_out()
        processed_df = pd.DataFrame(processed_input, columns=expected_cols)

        # Ensure all expected features are there (fill missing with 0)
        full_input = processed_df.reindex(columns=expected_cols, fill_value=0)

        # Predict
        prediction = model.predict(full_input)

        # Result
        st.subheader("Prediction Result")
        st.success("✅ Diabetic" if prediction[0] == 1 else "🟢 Not Diabetic")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write(traceback.format_exc())
