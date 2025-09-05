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
    try:
        model = joblib.load('optimized_random_forest.pkl')
        scaler = joblib.load('scaler.pkl')
        power_transformer = joblib.load('power_transformer.pkl')
        with open('selected_features.json', 'r') as f:
            selected_features = json.load(f)
        return model, scaler, power_transformer, selected_features
    except:
        return None, None, None, None

model, scaler, power_transformer, selected_features = load_model()

# App title and description
st.title("🩺 Diabetes Prediction App")
st.markdown("""
This app predicts the likelihood of diabetes based on health metrics.
Enter your health information below and click 'Predict' to see the results.
""")

# Show warning if model files aren't loaded
if model is None:
    st.error("""
    ❌ Model files not found. Please make sure you have these files in the same directory:
    - optimized_random_forest.pkl
    - scaler.pkl  
    - power_transformer.pkl
    - selected_features.json
    """)
    st.stop()

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
    age = st.slider("Age", 20, 100, 30)

with col2:
    st.header("Family History Assessment")
    
    st.write("Select all family members who have been diagnosed with diabetes:")
    
    # Family history checkboxes
    col2a, col2b = st.columns(2)
    
    with col2a:
        st.subheader("Immediate Family")
        mother = st.checkbox("Mother")
        father = st.checkbox("Father") 
        sister = st.checkbox("Sister")
        brother = st.checkbox("Brother")
    
    with col2b:
        st.subheader("Extended Family")
        maternal_grandmother = st.checkbox("Maternal Grandmother")
        maternal_grandfather = st.checkbox("Maternal Grandfather")
        paternal_grandmother = st.checkbox("Paternal Grandmother")
        paternal_grandfather = st.checkbox("Paternal Grandfather")
        other_relative = st.checkbox("Other close relative (aunt/uncle)")

    # Calculate Diabetes Pedigree Function based on family history
    dpf_score = 0.0

    # Immediate family (higher weight)
    if mother or father:
        dpf_score += 0.8  # Strong genetic link from parents
    if sister or brother:
        dpf_score += 0.6  # Siblings also indicate strong genetic risk

    # Grandparents (medium weight)
    grandparents_count = sum([maternal_grandmother, maternal_grandfather, 
                             paternal_grandmother, paternal_grandfather])
    dpf_score += grandparents_count * 0.3

    # Other relatives (small weight)
    if other_relative:
        dpf_score += 0.2

    # Additional risk for multiple affected immediate family members
    immediate_count = sum([mother, father, sister, brother])
    if immediate_count >= 2:
        dpf_score += 0.4  # Strong family pattern

    # Cap the score at 2.5 (maximum in the original dataset)
    dpf_score = min(dpf_score, 2.5)
    
    # Ensure minimum score is 0.0
    dpf_score = max(dpf_score, 0.0)

    # Display the calculated score with interpretation
    st.subheader("Genetic Risk Assessment")
    st.info(f"**Calculated Genetic Risk Score: {dpf_score:.2f}/2.50**")
    
    # Interpretation guide
    if dpf_score < 0.5:
        st.write("📊 **Interpretation:** Low genetic risk")
    elif dpf_score < 1.2:
        st.write("📊 **Interpretation:** Moderate genetic risk")
    elif dpf_score < 2.0:
        st.write("📊 **Interpretation:** High genetic risk")
    else:
        st.write("📊 **Interpretation:** Very high genetic risk")

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
    'DiabetesPedigreeFunction': dpf_score,  # Using calculated score
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
        
        # Create a visual risk meter
        risk_percentage = prediction_proba[0][1] * 100
        
        if prediction[0] == 1:
            st.error(f"🟥 **High risk of diabetes ({risk_percentage:.1f}% probability)**")
            st.warning("""
            **Recommendations:**
            - Please consult with a healthcare professional for further evaluation
            - Consider getting a HbA1c test for accurate diagnosis
            - Monitor your blood sugar levels regularly
            """)
        else:
            st.success(f"🟩 **Low risk of diabetes ({100-risk_percentage:.1f}% probability)**")
            st.info("""
            **Recommendations:**
            - Maintain a healthy lifestyle with regular exercise and balanced diet
            - Continue with regular health checkups
            - Be aware of diabetes symptoms and risk factors
            """)
        
        # Show probability breakdown with a gauge chart
        st.write("**Risk Probability Breakdown:**")
        prob_data = pd.DataFrame({
            'Risk Level': ['Low Risk', 'High Risk'],
            'Probability': [prediction_proba[0][0] * 100, prediction_proba[0][1] * 100]
        })
        
        # Display as a bar chart
        st.bar_chart(prob_data.set_index('Risk Level'))
        
        # Show key contributing factors
        st.subheader("Key Contributing Factors")
        
        factors = []
        if glucose > 125:
            factors.append(f"High glucose level ({glucose} mg/dL)")
        if bmi > 30:
            factors.append(f"High BMI ({bmi}) - Obese range")
        elif bmi > 25:
            factors.append(f"Elevated BMI ({bmi}) - Overweight range")
        if dpf_score > 1.0:
            factors.append(f"Strong family history (score: {dpf_score:.2f})")
        if age > 45:
            factors.append(f"Age ({age} - increased risk category)")
        
        if factors:
            for factor in factors:
                st.write(f"• {factor}")
        else:
            st.write("• No major risk factors identified from your inputs")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.info("Please make sure all required model files are in the same directory.")

# Add some information about diabetes
with st.expander("ℹ️ About Diabetes and Risk Factors"):
    st.markdown("""
    **Diabetes** is a chronic condition that affects how your body turns food into energy.
    
    **Key Risk Factors:**
    - **High glucose levels** (≥126 mg/dL when fasting)
    - **Obesity or high BMI** (BMI ≥30 significantly increases risk)
    - **Family history** of diabetes (immediate relatives)
    - **High blood pressure** (≥140/90 mmHg)
    - **Age** (risk increases significantly after 45 years)
    - **Gestational diabetes** during pregnancy
    
    **Normal Health Ranges:**
    - Glucose: <100 mg/dL (fasting)
    - BMI: 18.5-24.9
    - Blood Pressure: <120/80 mmHg
    - Insulin: 2.6-24.9 μU/mL
    
    **Disclaimer:** This prediction is based on a machine learning model and should not replace 
    professional medical advice. Always consult with healthcare professionals for proper diagnosis.
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Model: Optimized Random Forest")
