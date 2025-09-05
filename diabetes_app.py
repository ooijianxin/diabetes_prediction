import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import StandardScaler, PowerTransformer

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model and preprocessing objects
@st.cache_resource
def load_model():
    model = joblib.load('optimized_random_forest.pkl')
    scaler = joblib.load('scaler.pkl')
    power_transformer = joblib.load('power_transformer.pkl')
    with open('selected_features.json', 'r') as f:
        selected_features = json.load(f)
    return model, scaler, power_transformer, selected_features

try:
    model, scaler, power_transformer, selected_features = load_model()
except:
    st.error("❌ Model files not found. Please make sure you have the trained model files.")
    st.stop()

# App title and description
st.title("🩺 Diabetes Prediction App")
st.markdown("""
This app predicts the likelihood of diabetes based on health metrics.
Enter your health information below and click 'Predict' to see the results.
""")

# Create two columns for layout
col1, col2 = st.columns(2)

with col1:
    st.header("Patient Information")
    
    # Input fields for the user
    pregnancies = st.slider("Number of Pregnancies", 0, 20, 0)
    glucose = st.slider("Glucose Level (mg/dL)", 0, 200, 100)
    blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 130, 70)
    skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)
    insulin = st.slider("Insulin Level (mu U/ml)", 0, 850, 80)
    bmi = st.slider("BMI", 0.0, 70.0, 25.0, 0.1)
    diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, 0.01)
    age = st.slider("Age", 20, 100, 30)

with col2:
    st.header("Additional Features")
    
    # Calculate derived features (same as in your training)
    glucose_bmi = glucose * bmi
    age_glucose = age * glucose
    bp_bmi = blood_pressure * bmi
    insulin_glucose = insulin * glucose
    glucose_insulin_ratio = glucose / (insulin + 1) if insulin != 0 else glucose
    bmi_age_ratio = bmi / age if age != 0 else bmi
    preg_age_ratio = pregnancies / (age + 1) if age != 0 else pregnancies
    glucose_sq = glucose ** 2
    bmi_sq = bmi ** 2
    age_sq = age ** 2
    metabolic_risk = (glucose / 100) + (bmi / 30) + (age / 50)
    insulin_resistance = (glucose * insulin) / 405 if insulin != 0 else 0
    
    # Display calculated features (read-only)
    st.info("Calculated Features:")
    st.write(f"Glucose × BMI: {glucose_bmi:.2f}")
    st.write(f"Age × Glucose: {age_glucose:.2f}")
    st.write(f"Metabolic Risk Score: {metabolic_risk:.2f}")
    
    # Add categorical features (dummy variables)
    glucose_level = "Normal"
    if glucose > 300:
        glucose_level = "Severe"
    elif glucose > 125:
        glucose_level = "Diabetic"
    elif glucose > 100:
        glucose_level = "Prediabetic"
    
    bmi_category = "Normal"
    if bmi > 30:
        bmi_category = "Obese"
    elif bmi > 25:
        bmi_category = "Overweight"
    elif bmi < 18.5:
        bmi_category = "Underweight"
    
    age_group = "20-30"
    if age > 50:
        age_group = "50+"
    elif age > 40:
        age_group = "40-50"
    elif age > 30:
        age_group = "30-40"

# Create a dictionary with all features
input_data = {
    'Pregnancies': pregnancies,
    'Glucose': glucose,
    'BloodPressure': blood_pressure,
    'SkinThickness': skin_thickness,
    'Insulin': insulin,
    'BMI': bmi,
    'DiabetesPedigreeFunction': diabetes_pedigree,
    'Age': age,
    'Glucose_BMI': glucose_bmi,
    'Age_Glucose': age_glucose,
    'BP_BMI': bp_bmi,
    'Insulin_Glucose': insulin_glucose,
    'Glucose_Insulin_Ratio': glucose_insulin_ratio,
    'BMI_Age_Ratio': bmi_age_ratio,
    'Preg_Age_Ratio': preg_age_ratio,
    'Glucose_sq': glucose_sq,
    'BMI_sq': bmi_sq,
    'Age_sq': age_sq,
    'Metabolic_Risk': metabolic_risk,
    'Insulin_Resistance': insulin_resistance,
    'Glucose_Level_Prediabetic': 1 if glucose_level == "Prediabetic" else 0,
    'Glucose_Level_Diabetic': 1 if glucose_level == "Diabetic" else 0,
    'Glucose_Level_Severe': 1 if glucose_level == "Severe" else 0,
    'BMI_Category_Overweight': 1 if bmi_category == "Overweight" else 0,
    'BMI_Category_Obese': 1 if bmi_category == "Obese" else 0,
    'BMI_Category_Underweight': 1 if bmi_category == "Underweight" else 0,
    'Age_Group_30-40': 1 if age_group == "30-40" else 0,
    'Age_Group_40-50': 1 if age_group == "40-50" else 0,
    'Age_Group_50+': 1 if age_group == "50+" else 0
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Filter to only include the selected features
input_df_selected = input_df[selected_features]

# Prediction button
if st.button("Predict Diabetes Risk", type="primary"):
    try:
        # Preprocess the input data
        input_scaled = scaler.transform(input_df_selected)
        input_transformed = power_transformer.transform(input_scaled)
        
        # Make prediction
        prediction = model.predict(input_transformed)
        prediction_proba = model.predict_proba(input_transformed)
        
        # Display results
        st.subheader("Prediction Results")
        
        if prediction[0] == 1:
            st.error(f"🟥 High risk of diabetes ({prediction_proba[0][1]*100:.2f}% probability)")
            st.warning("Please consult with a healthcare professional for further evaluation.")
        else:
            st.success(f"🟩 Low risk of diabetes ({prediction_proba[0][0]*100:.2f}% probability)")
            st.info("Maintain a healthy lifestyle with regular exercise and balanced diet.")
        
        # Show probability breakdown
        st.write("Probability Breakdown:")
        prob_df = pd.DataFrame({
            'Class': ['No Diabetes', 'Diabetes'],
            'Probability': [prediction_proba[0][0], prediction_proba[0][1]]
        })
        st.bar_chart(prob_df.set_index('Class'))
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Add some information about diabetes
with st.expander("ℹ️ About Diabetes and Risk Factors"):
    st.markdown("""
    **Diabetes** is a chronic condition that affects how your body turns food into energy.
    
    **Key Risk Factors:**
    - High glucose levels
    - Obesity or high BMI
    - Family history of diabetes
    - High blood pressure
    - Age (risk increases with age)
    - Gestational diabetes during pregnancy
    
    **Disclaimer:** This prediction is based on a machine learning model and should not replace 
    professional medical advice. Always consult with healthcare professionals for proper diagnosis.
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Model: Optimized Random Forest")