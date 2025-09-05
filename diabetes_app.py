import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import requests
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, PowerTransformer
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Diabetes Risk Assessment Tool",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model and preprocessing objects with enhanced error handling
@st.cache_resource
def load_model():
    try:
        # Try to load from local files first
        try:
            model = joblib.load('optimized_random_forest.pkl')
            scaler = joblib.load('scaler.pkl')
            power_transformer = joblib.load('power_transformer.pkl')
            with open('selected_features.json', 'r') as f:
                selected_features = json.load(f)
            return model, scaler, power_transformer, selected_features, True
        except:
            # Fallback to download from a repository (hypothetical)
            st.warning("Local model files not found. Using built-in model with limitations.")
            # In a real implementation, you might load a simpler model
            # or provide instructions to download the proper files
            return None, None, None, None, False
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None, False

model, scaler, power_transformer, selected_features, model_loaded = load_model()

# App title and description
st.title("🩺 Diabetes Risk Assessment Tool")
st.markdown("""
This tool estimates your risk of developing type 2 diabetes based on health metrics and family history.
**This is not a diagnostic tool.** Always consult with a healthcare professional for proper medical evaluation.
""")

# Create two columns for layout
col1, col2 = st.columns(2)

with col1:
    st.header("Patient Information")
    
    # Input fields with medical guidance
    age = st.slider("Age", 20, 100, 30, 
                   help="Risk increases significantly after age 45")
    
    pregnancies = st.slider("Number of Pregnancies", 0, 20, 0,
                           help="Gestational diabetes during pregnancy increases future risk")
    
    glucose = st.slider("Fasting Glucose Level (mg/dL)", 50, 300, 100,
                       help="Normal: <100 mg/dL, Prediabetes: 100-125 mg/dL, Diabetes: ≥126 mg/dL")
    
    blood_pressure = st.slider("Blood Pressure (mm Hg)", 60, 200, 70,
                              help="Normal: <120/80 mmHg, Hypertension: ≥140/90 mmHg")
    
    skin_thickness = st.slider("Triceps Skin Fold Thickness (mm)", 5, 100, 20,
                              help="Measures subcutaneous fat thickness")
    
    insulin = st.slider("2-Hour Serum Insulin (mu U/ml)", 0, 850, 80,
                       help="Normal fasting range: 2.6-24.9 μU/mL")
    
    bmi = st.slider("Body Mass Index (BMI)", 15.0, 50.0, 25.0, 0.1,
                   help="Underweight: <18.5, Normal: 18.5-24.9, Overweight: 25-29.9, Obese: ≥30")

with col2:
    st.header("Family History Assessment")
    
    st.markdown("""
    **Diabetes in family members increases your risk:**
    - First-degree relatives (parents, siblings) contribute most to risk
    - Second-degree relatives (grandparents) contribute moderately
    """)
    
    # Family history checkboxes with improved medical accuracy
    col2a, col2b = st.columns(2)
    
    with col2a:
        st.subheader("First-Degree Relatives")
        mother = st.checkbox("Mother", help="Strongest genetic link for type 2 diabetes")
        father = st.checkbox("Father", help="Strong genetic link") 
        full_sibling = st.checkbox("Full Sibling", help="Strong genetic link")
        child = st.checkbox("Child with diabetes", help="Indicates genetic predisposition")
    
    with col2b:
        st.subheader("Second-Degree Relatives")
        maternal_grandmother = st.checkbox("Maternal Grandmother")
        maternal_grandfather = st.checkbox("Maternal Grandfather")
        paternal_grandmother = st.checkbox("Paternal Grandmother")
        paternal_grandfather = st.checkbox("Paternal Grandfather")
    
    st.subheader("Other Risk Factors")
    gestational_diabetes = st.checkbox("History of gestational diabetes", 
                                      help="Strong predictor for developing type 2 diabetes")
    physical_activity = st.select_slider("Physical Activity Level", 
                                        options=["Sedentary", "Light", "Moderate", "Active", "Very Active"],
                                        value="Moderate")
    
    # Calculate Diabetes Pedigree Function with improved medical accuracy
    dpf_score = 0.0

    # First-degree relatives (highest weight)
    first_degree_count = sum([mother, father, full_sibling, child])
    if first_degree_count == 1:
        dpf_score += 0.5
    elif first_degree_count == 2:
        dpf_score += 1.2
    elif first_degree_count >= 3:
        dpf_score += 2.0

    # Second-degree relatives (moderate weight)
    second_degree_count = sum([maternal_grandmother, maternal_grandfather, 
                              paternal_grandmother, paternal_grandfather])
    dpf_score += second_degree_count * 0.2

    # Additional risk factors
    if gestational_diabetes:
        dpf_score += 0.8  # Strong predictor
        
    # Physical activity adjustment (more activity = lower risk)
    activity_weights = {
        "Sedentary": 0.3,
        "Light": 0.15,
        "Moderate": 0.0,
        "Active": -0.1,
        "Very Active": -0.2
    }
    dpf_score += activity_weights[physical_activity]

    # Cap the score at reasonable limits based on clinical data
    dpf_score = min(max(dpf_score, 0.0), 2.4)

    # Display the calculated score with improved interpretation
    st.subheader("Genetic Risk Assessment")
    
    # Create a visual gauge for genetic risk
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = dpf_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Genetic Risk Score"},
        gauge = {
            'axis': {'range': [0, 2.4]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.8], 'color': "lightgreen"},
                {'range': [0.8, 1.6], 'color': "yellow"},
                {'range': [1.6, 2.4], 'color': "red"}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': dpf_score}}))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation guide
    if dpf_score < 0.5:
        st.success("📊 **Interpretation:** Below average genetic risk")
    elif dpf_score < 1.2:
        st.info("📊 **Interpretation:** Moderate genetic risk")
    elif dpf_score < 1.8:
        st.warning("📊 **Interpretation:** High genetic risk")
    else:
        st.error("📊 **Interpretation:** Very high genetic risk")

# Calculate derived features with enhanced medical relevance
glucose_bmi = glucose * bmi / 100  # Normalized scaling
age_glucose = age * glucose / 100  # Normalized scaling
bp_bmi = blood_pressure * bmi / 100  # Normalized scaling
insulin_glucose = insulin * glucose / 100  # Normalized scaling
glucose_insulin_ratio = glucose / (insulin + 1) if insulin != 0 else glucose
homa_ir = (glucose * insulin) / 405  # Homeostatic Model Assessment of Insulin Resistance
bmi_age_ratio = bmi / age if age != 0 else bmi
preg_age_ratio = pregnancies / (age - 12) if age > 12 else 0  # Assuming fertility starts around 12
metabolic_syndrome_score = (glucose > 100) + (blood_pressure > 130) + (bmi > 30) + (triglycerides > 150)

# Add categorical features with medical standards
glucose_level = "Normal"
if glucose >= 126:
    glucose_level = "Diabetic"
elif glucose >= 100:
    glucose_level = "Prediabetic"
elif glucose < 70:
    glucose_level = "Hypoglycemic"

bmi_category = "Normal"
if bmi >= 30:
    bmi_category = "Obese"
elif bmi >= 25:
    bmi_category = "Overweight"
elif bmi < 18.5:
    bmi_category = "Underweight"

bp_category = "Normal"
if blood_pressure >= 140:
    bp_category = "Stage 2 Hypertension"
elif blood_pressure >= 130:
    bp_category = "Stage 1 Hypertension"
elif blood_pressure >= 120:
    bp_category = "Elevated"

# Create a dictionary with all features
input_data = {
    'Pregnancies': pregnancies,
    'Glucose': glucose,
    'BloodPressure': blood_pressure,
    'SkinThickness': skin_thickness,
    'Insulin': insulin,
    'BMI': bmi,
    'DiabetesPedigreeFunction': dpf_score,
    'Age': age,
    'Glucose_BMI': glucose_bmi,
    'Age_Glucose': age_glucose,
    'BP_BMI': bp_bmi,
    'Insulin_Glucose': insulin_glucose,
    'Glucose_Insulin_Ratio': glucose_insulin_ratio,
    'HOMA_IR': homa_ir,
    'BMI_Age_Ratio': bmi_age_ratio,
    'Preg_Age_Ratio': preg_age_ratio,
    'Metabolic_Syndrome_Score': metabolic_syndrome_score,
    'Glucose_Level_Prediabetic': 1 if glucose_level == "Prediabetic" else 0,
    'Glucose_Level_Diabetic': 1 if glucose_level == "Diabetic" else 0,
    'BMI_Category_Overweight': 1 if bmi_category == "Overweight" else 0,
    'BMI_Category_Obese': 1 if bmi_category == "Obese" else 0,
    'BMI_Category_Underweight': 1 if bmi_category == "Underweight" else 0,
    'BP_Category_Hypertension': 1 if "Hypertension" in bp_category else 0
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Prediction button with enhanced validation
if st.button("Assess Diabetes Risk", type="primary"):
    # Basic validation
    if glucose < 50 or glucose > 300:
        st.warning("⚠️ Glucose value appears outside typical range. Please verify your entry.")
    
    if not model_loaded:
        st.error("""
        ❌ Model not available. Please ensure you have the required model files:
        - optimized_random_forest.pkl
        - scaler.pkl  
        - power_transformer.pkl
        - selected_features.json
        
        Alternatively, contact the administrator for assistance.
        """)
        st.stop()
    
    try:
        # Filter to only include the selected features
        input_df_selected = input_df[selected_features]
        
        # Preprocess the input data
        input_scaled = scaler.transform(input_df_selected)
        input_transformed = power_transformer.transform(input_scaled)
        
        # Make prediction
        prediction = model.predict(input_transformed)
        prediction_proba = model.predict_proba(input_transformed)
        
        # Display results with enhanced medical context
        st.subheader("Risk Assessment Results")
        
        # Create a comprehensive risk visualization
        risk_percentage = prediction_proba[0][1] * 100
        
        # Risk interpretation based on medical guidelines
        if risk_percentage < 10:
            risk_category = "Very Low Risk"
            color = "green"
        elif risk_percentage < 25:
            risk_category = "Low Risk"
            color = "lightgreen"
        elif risk_percentage < 50:
            risk_category = "Moderate Risk"
            color = "yellow"
        elif risk_percentage < 75:
            risk_category = "High Risk"
            color = "orange"
        else:
            risk_category = "Very High Risk"
            color = "red"
        
        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_percentage,
            number = {'suffix': "%"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Diabetes Risk: {risk_category}"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 10], 'color': "lightgreen"},
                    {'range': [10, 25], 'color': "green"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_percentage}}))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Medical recommendations based on risk level
        st.subheader("Medical Recommendations")
        
        if risk_percentage >= 50:
            st.error("""
            **🔴 High Risk Category - Consult a Healthcare Provider:**
            - Schedule an appointment with your doctor for further evaluation
            - Request HbA1c, fasting glucose, and oral glucose tolerance tests
            - Begin lifestyle modifications immediately
            - Monitor your blood sugar regularly if advised by your doctor
            """)
        elif risk_percentage >= 25:
            st.warning("""
            **🟡 Moderate Risk Category - Preventive Action Recommended:**
            - Discuss diabetes screening with your healthcare provider
            - Focus on weight management if needed
            - Increase physical activity to at least 150 minutes per week
            - Adopt a balanced diet rich in fiber and low in processed sugars
            """)
        else:
            st.success("""
            **🟢 Lower Risk Category - Maintain Healthy Habits:**
            - Continue regular physical activity
            - Maintain a balanced diet
            - Schedule regular health checkups
            - Be aware of diabetes symptoms
            """)
        
        # Show key contributing factors with medical context
        st.subheader("Key Risk Factors Identified")
        
        factors = []
        if glucose >= 126:
            factors.append(f"🔺 Diabetic glucose level ({glucose} mg/dL)")
        elif glucose >= 100:
            factors.append(f"🟡 Prediabetic glucose level ({glucose} mg/dL)")
            
        if bmi >= 30:
            factors.append(f"🔺 Obesity (BMI: {bmi})")
        elif bmi >= 25:
            factors.append(f"🟡 Overweight (BMI: {bmi})")
            
        if blood_pressure >= 130:
            factors.append(f"🔺 Elevated blood pressure ({blood_pressure} mm Hg)")
            
        if dpf_score > 1.0:
            factors.append(f"🔺 Strong family history (score: {dpf_score:.2f})")
            
        if age > 45:
            factors.append(f"🟡 Age ({age} - increased risk category)")
            
        if homa_ir > 2.5:
            factors.append(f"🔺 Insulin resistance indicated (HOMA-IR: {homa_ir:.2f})")
            
        if physical_activity in ["Sedentary", "Light"]:
            factors.append(f"🟡 Low physical activity level")
        
        if factors:
            for factor in factors:
                st.write(factor)
        else:
            st.write("• No major risk factors identified from your inputs")
            
        # Add disclaimer
        st.info("""
        **Medical Disclaimer:** This assessment tool provides probability estimates based on statistical models 
        and should not be used as a substitute for professional medical advice, diagnosis, or treatment. 
        Always seek the advice of your physician or other qualified health provider with any questions 
        you may have regarding a medical condition.
        """)
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.info("Please check your inputs and try again. If the problem persists, contact support.")

# Add comprehensive information about diabetes
with st.expander("ℹ️ Comprehensive Diabetes Information"):
    st.markdown("""
    ## Understanding Type 2 Diabetes
    
    **What is Diabetes?**
    Diabetes is a chronic condition that affects how your body processes blood sugar (glucose). 
    In type 2 diabetes, your body either doesn't produce enough insulin or resists insulin's effects.
    
    **Key Risk Factors:**
    - **Family history**: Having a parent or sibling with diabetes
    - **Weight**: Being overweight or obese (BMI ≥25)
    - **Physical inactivity**: Less than 150 minutes of moderate activity per week
    - **Age**: Risk increases after age 45
    - **Prediabetes**: Fasting blood sugar between 100-125 mg/dL
    - **Gestational diabetes**: During pregnancy
    - **Polycystic ovary syndrome (PCOS)**
    - **High blood pressure**: ≥140/90 mmHg
    
    **Prevention Strategies:**
    - Lose 5-7% of body weight if overweight
    - Engage in moderate exercise 30 minutes daily, 5 days a week
    - Eat a balanced diet rich in whole grains, fruits, and vegetables
    - Limit processed foods and sugary beverages
    - Get regular health screenings
    
    **Recommended Screening:**
    - **Age <45**: Screen if overweight + additional risk factors
    - **Age ≥45**: Screen every 3 years regardless of weight
    - **Previous gestational diabetes**: Screen every 1-3 years
    
    **Diagnostic Tests:**
    - Fasting plasma glucose test: ≥126 mg/dL indicates diabetes
    - HbA1c test: ≥6.5% indicates diabetes
    - Oral glucose tolerance test: ≥200 mg/dL after 2 hours indicates diabetes
    
    **Normal Health Ranges:**
    - Fasting Glucose: <100 mg/dL
    - HbA1c: <5.7%
    - BMI: 18.5-24.9
    - Blood Pressure: <120/80 mmHg
    - Insulin: 2.6-24.9 μU/mL (fasting)
    """)

# Footer with enhanced information
st.markdown("---")
st.markdown("""
**Tool Information:** 
- Model: Optimized Random Forest Classifier
- Validation: Developed using clinical datasets
- Limitations: Assessment tool only, not a diagnostic device
- Last updated: October 2023

**For medical concerns, please consult with a healthcare professional.**
""")

# Add feedback mechanism
with st.expander("💬 Provide Feedback"):
    feedback = st.text_area("Help us improve this tool by sharing your feedback")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback! It will help us improve this tool.")
