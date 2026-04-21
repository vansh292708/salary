
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the model components
try:
    with open('svm_model_components.pkl', 'rb') as file:
        model_components = pickle.load(file)
    svm_model = model_components['svm_model']
    scaler_X = model_components['scaler_X']
    scaler_y = model_components['scaler_y']
    label_encoders = model_components['label_encoders']
except FileNotFoundError:
    st.error("Error: 'svm_model_components.pkl' not found. Please ensure the model components are saved.")
    st.stop()

# --- Streamlit App ----
st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("Salary Prediction with SVM")
st.write("Enter the details below to predict the salary.")

# Input fields for features
st.header("Job Details")

rating = st.slider("Rating", min_value=1.0, max_value=5.0, value=3.5, step=0.1)

# For categorical features, we need to handle them carefully.
# Ideally, the Streamlit app would have dropdowns populated by unique values from the training data.
# For simplicity, we'll ask for text input, but this assumes the user provides valid, known categories.
# In a real-world scenario, you'd extract unique values from your original dataset to create these options.

# Placeholder for unique values - In a real app, these would be loaded from a config or pre-processed data
# Since we don't have access to the original categorical values, we'll prompt for direct encoded values or use a simpler approach.
# Let's assume for now, we're providing a list of most common ones or using an example.

# To make this robust, we should map the user input to the label encoder's known classes.
# This would require saving the class list for each LabelEncoder.
# For this demonstration, we'll assume the user provides a string that exists in the encoder's classes.

# For `Company Name`, `Job Title`, `Location`, `Employment Status`, `Job Roles`
# We need the original unique values to create dropdowns for the user.
# Since `label_encoders` contains fitted encoders, we can get their classes.

company_name_options = list(label_encoders['Company Name'].classes_)
job_title_options = list(label_encoders['Job Title'].classes_)
location_options = list(label_encoders['Location'].classes_)
employment_status_options = list(label_encoders['Employment Status'].classes_)
jop_roles_options = list(label_encoders['Job Roles'].classes_)

company_name_input = st.selectbox("Company Name", options=company_name_options)
job_title_input = st.selectbox("Job Title", options=job_title_options)
salaries_reported = st.number_input("Salaries Reported", min_value=1, value=1)
location_input = st.selectbox("Location", options=location_options)
employment_status_input = st.selectbox("Employment Status", options=employment_status_options)
jop_roles_input = st.selectbox("Job Roles", options=jop_roles_options)

# Preprocess inputs
def preprocess_input(rating, company_name_input, job_title_input, salaries_reported, location_input, employment_status_input, jop_roles_input):
    # Create a DataFrame for the input
    input_df = pd.DataFrame([[rating, company_name_input, job_title_input, salaries_reported, location_input, employment_status_input, jop_roles_input]],
                            columns=['Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles'])

    # Apply Label Encoding using the loaded encoders
    for col in ['Company Name', 'Job Title', 'Location', 'Employment Status', 'Job Roles']:
        if col in label_encoders:
            # Handle unseen labels by assigning a default value (e.g., -1 or the most frequent)
            # For simplicity here, we assume inputs will be among known classes
            try:
                input_df[col] = label_encoders[col].transform(input_df[col])
            except ValueError:
                st.warning(f"Categorical value for '{col}' not recognized. Using a default encoded value.")
                input_df[col] = -1 # Or a more sophisticated handling

    # Scale numerical features (all features are numerical after encoding for X_train_scaled)
    # The order of columns in X_train is crucial here.
    # Ensure input_df columns match the order X was trained on.
    # X.columns: Index(['Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles'], dtype='object')

    # Reorder columns to match the training data's X
    X_cols = ['Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles']
    processed_input = scaler_X.transform(input_df[X_cols])

    return processed_input

if st.button("Predict Salary"):
    processed_input = preprocess_input(rating, company_name_input, job_title_input, salaries_reported, location_input, employment_status_input, jop_roles_input)

    # Make prediction
    predicted_salary_scaled = svm_model.predict(processed_input)

    # Inverse transform the scaled prediction to get actual salary
    predicted_salary = scaler_y.inverse_transform(predicted_salary_scaled.reshape(-1, 1))[0][0]

    st.success(f"Predicted Salary: ₹{predicted_salary:,.2f}")

st.markdown("--- Developed for educational purposes --- ")
