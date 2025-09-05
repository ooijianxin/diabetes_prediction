import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import StandardScaler, PowerTransformer

# Set page configuration
st.set_page_config(
    page_title="Diabetes Risk Assessment",
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
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None, None, None, None

model, scaler, power_transformer, selected_features = load_model()

# App title and description
st.title("🩺 Diabetes Risk Assessment Tool")
st.markdown("""
This tool assesses your risk of developing diabetes based on health metrics and family history.
**This is not a diagnostic tool.** Please consult a healthcare professional for proper medical advice.
""")

# Show warning if model files aren't loaded
if model is None:
    st.error("""
    ❌ Required model files not found. Please make sure you have these files in the same directory:
    - optimized_random_forest.pkl
    - scaler.pkl  
    - power_transformer.pkl
    - selected_features.json
    """)
    st.stop()

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["Health Assessment", "Results", "About Diabetes"])

with tab1:
    # Create two columns for layout
    col1, col2 = st.columns(2)

    with col1:
        st.header("Basic Health Information")
        
        # Input fields for the user with better defaults and ranges
        age = st.slider("Age (years)", 20, 100, 35, 
                       help="Risk increases significantly after age 45")
        
        pregnancies = st.slider("Number of Pregnancies", 0, 20, 0,
                               help="Multiple pregnancies can increase diabetes risk")
        
        glucose = st.slider("Fasting Glucose Level (mg/dL)", 50, 300, 100,
                           help="Normal fasting glucose is below 100 mg/dL")
        
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 60, 180, 80,
                                  help="Normal blood pressure is below 120/80 mmHg")
        
        st.info("""
        **Normal Health Ranges:**
        - Glucose: <100 mg/dL (fasting)
        - Blood Pressure: <120/80 mmHg
        """)

    with col2:
        st.header("Additional Health Metrics")
        
        skin_thickness = st.slider("Skin Thickness (mm)", 5, 100, 25,
                                  help="Triceps skin fold thickness")
        
        insulin = st.slider("Insulin Level (μU/mL)", 0, 300, 50,
                           help="Normal fasting insulin: 2.6-24.9 μU/mL")
        
        bmi = st.slider("Body Mass Index (BMI)", 15.0, 50.0, 25.0, 0.1,
                       help="Normal BMI: 18.5-24.9")
        
        # Display BMI categories
        bmi_category = "Underweight" if bmi < 18.5 else \
                      "Normal" if bmi < 25 else \
                      "Overweight" if bmi < 30 else "Obese"
        st.write(f"**BMI Category:** {bmi_category}")
        
        st.info("""
        **BMI Categories:**
        - Underweight: <18.5
        - Normal: 18.5-24.9  
        - Overweight: 25-29.9
        - Obese: ≥30
        """)

    st.header("Family History Assessment")
    
    st.write("Select all family members who have been diagnosed with diabetes:")
    
    # Family history checkboxes in a more organized layout
    col_fam1, col_fam2, col_fam3 = st.columns(3)
    
    with col_fam1:
        st.subheader("Immediate Family")
        mother = st.checkbox("Mother")
        father = st.checkbox("Father") 
        siblings = st.checkbox("Siblings")
    
    with col_fam2:
        st.subheader("Grandparents")
        maternal_grandmother = st.checkbox("Maternal Grandmother")
        maternal_grandfather = st.checkbox("Maternal Grandfather")
        paternal_grandmother = st.checkbox("Paternal Grandmother")
        paternal_grandfather = st.checkbox("Paternal Grandfather")
    
    with col_fam3:
        st.subheader("Other Relatives")
        other_relative = st.checkbox("Other close relative (aunt/uncle/cousin)")
        gestational = st.checkbox("History of gestational diabetes")

    # Calculate Diabetes Pedigree Function based on family history
    # Using a more medically validated approach
    dpf_score = 0.0

    # Immediate family (highest weight)
    if mother or father:
        dpf_score += 0.5  # Each parent contributes significantly
    if siblings:
        dpf_score += 0.3  # Siblings also indicate genetic risk

    # Grandparents (medium weight)
    grandparents_count = sum([maternal_grandmother, maternal_grandfather, 
                             paternal_grandmother, paternal_grandfather])
    dpf_score += grandparents_count * 0.15

    # Other factors
    if other_relative:
        dpf_score += 0.1
    if gestational:
        dpf_score += 0.2

    # Cap the score at 2.5 (maximum in the original dataset)
    dpf_score = min(dpf_score, 2.5)
    
    # Ensure minimum score is 0.078 (minimum in original dataset)
    dpf_score = max(dpf_score, 0.078)

    # Display the calculated score with interpretation
    st.subheader("Genetic Risk Assessment")
    
    risk_color = "green" if dpf_score < 0.5 else \
                "orange" if dpf_score < 1.2 else \
                "red"
    
    st.markdown(f"**Calculated Genetic Risk Score: <span style='color:{risk_color};'>{dpf_score:.2f}/2.50</span>**", 
                unsafe_allow_html=True)
    
    # Interpretation guide
    if dpf_score < 0.5:
        st.write("📊 **Interpretation:** Low genetic risk")
    elif dpf_score < 1.2:
        st.write("📊 **Interpretation:** Moderate genetic risk")
    elif dpf_score < 2.0:
        st.write("📊 **Interpretation:** High genetic risk")
    else:
        st.write("📊 **Interpretation:** Very high genetic risk")

# Calculate derived features (consistent with training)
glucose_bmi = glucose * bmi
age_glucose = age * glucose
bp_bmi = blood_pressure * bmi
insulin_glucose = insulin * glucose
glucose_insulin_ratio = glucose / (insulin + 1e-5)  # Avoid division by zero
bmi_age_ratio = bmi / max(age, 1)  # Avoid division by zero
preg_age_ratio = pregnancies / max(age + 1, 1)  # Avoid division by zero
glucose_sq = glucose ** 2
bmi_sq = bmi ** 2
age_sq = age ** 2
metabolic_risk = (glucose / 100) + (bmi / 30) + (age / 50)
insulin_resistance = (glucose * insulin) / 405 if insulin > 0 else 0

# Categorical features (must match training exactly)
glucose_level_normal = 1 if glucose <= 100 else 0
glucose_level_prediabetic = 1 if 100 < glucose <= 125 else 0
glucose_level_diabetic = 1 if 125 < glucose <= 200 else 0
glucose_level_severe = 1 if glucose > 200 else 0

bmi_category_underweight = 1 if bmi < 18.5 else 0
bmi_category_normal = 1 if 18.5 <= bmi < 25 else 0
bmi_category_overweight = 1 if 25 <= bmi < 30 else 0
bmi_category_obese = 1 if bmi >= 30 else 0

age_group_20_30 = 1 if 20 <= age < 30 else 0
age_group_30_40 = 1 if 30 <= age < 40 else 0
age_group_40_50 = 1 if 40 <= age < 50 else 0
age_group_50_plus = 1 if age >= 50 else 0

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
    'BMI_Age_Ratio': bmi_age_ratio,
    'Preg_Age_Ratio': preg_age_ratio,
    'Glucose_sq': glucose_sq,
    'BMI_sq': bmi_sq,
    'Age_sq': age_sq,
    'Metabolic_Risk': metabolic_risk,
    'Insulin_Resistance': insulin_resistance,
    'Glucose_Level_Normal': glucose_level_normal,
    'Glucose_Level_Prediabetic': glucose_level_prediabetic,
    'Glucose_Level_Diabetic': glucose_level_diabetic,
    'Glucose_Level_Severe': glucose_level_severe,
    'BMI_Category_Underweight': bmi_category_underweight,
    'BMI_Category_Normal': bmi_category_normal,
    'BMI_Category_Overweight': bmi_category_overweight,
    'BMI_Category_Obese': bmi_category_obese,
    'Age_Group_20-30': age_group_20_30,
    'Age_Group_30-40': age_group_30_40,
    'Age_Group_40-50': age_group_40_50,
    'Age_Group_50+': age_group_50_plus
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Ensure we have all the selected features
missing_features = set(selected_features) - set(input_df.columns)
if missing_features:
    st.error(f"Missing features: {missing_features}")
    # Add missing features with default value of 0
    for feature in missing_features:
        input_df[feature] = 0

# Filter to only include the selected features
input_df_selected = input_df[selected_features]

# Prediction button
if st.button("Assess Diabetes Risk", type="primary", use_container_width=True):
    with tab2:
        try:
            # Preprocess the input data
            input_scaled = scaler.transform(input_df_selected)
            input_transformed = power_transformer.transform(input_scaled)
            
            # Make prediction
            prediction = model.predict(input_transformed)
            prediction_proba = model.predict_proba(input_transformed)
            
            # Display results
            st.header("Risk Assessment Results")
            
            # Create a visual risk meter
            risk_percentage = prediction_proba[0][1] * 100
            
            # Create a more detailed risk assessment
            if risk_percentage < 20:
                risk_level = "Very Low Risk"
                color = "green"
                icon = "🟢"
            elif risk_percentage < 40:
                risk_level = "Low Risk"
                color = "green"
                icon = "🟢"
            elif risk_percentage < 60:
                risk_level = "Moderate Risk"
                color = "orange"
                icon = "🟡"
            elif risk_percentage < 80:
                risk_level = "High Risk"
                color = "red"
                icon = "🟠"
            else:
                risk_level = "Very High Risk"
                color = "darkred"
                icon = "🔴"
            
            # Display risk level
            st.markdown(f"### {icon} {risk_level} - {risk_percentage:.1f}% probability")
            st.markdown(f"<h3 style='color:{color};'>{risk_level} - {risk_percentage:.1f}% probability</h3>", 
                       unsafe_allow_html=True)
            
            # Show probability breakdown with a gauge chart
            st.subheader("Risk Probability Breakdown")
            
            # Create a visual gauge
            gauge_html = f"""
            <div style="width: 100%; background: #f0f0f0; border-radius: 10px; padding: 3px;">
                <div style="width: {risk_percentage}%; background: {color}; 
                         color: white; text-align: center; border-radius: 10px; 
                         padding: 5px; transition: width 0.5s;">
                    {risk_percentage:.1f}%
                </div>
            </div>
            """
            st.markdown(gauge_html, unsafe_allow_html=True)
            
            # Recommendations based on risk level
            st.subheader("Recommendations")
            
            if risk_percentage < 40:
                st.success("""
                **Maintenance Plan:**
                - Continue with your healthy lifestyle habits
                - Maintain regular physical activity (150+ minutes/week)
                - Eat a balanced diet rich in fruits, vegetables, and whole grains
                - Schedule annual health check-ups
                """)
            elif risk_percentage < 60:
                st.warning("""
                **Prevention Plan:**
                - Consult with a healthcare provider for personalized advice
                - Increase physical activity to 30 minutes most days
                - Focus on weight management if needed
                - Reduce intake of processed foods and sugars
                - Consider getting a HbA1c test for baseline measurement
                """)
            else:
                st.error("""
                **Action Plan:**
                - **Schedule an appointment with a healthcare professional promptly**
                - Request comprehensive diabetes screening (fasting glucose, HbA1c)
                - Implement lifestyle changes immediately
                - Monitor your blood sugar levels regularly if advised
                - Join a diabetes prevention program if available
                """)
            
            # Show key contributing factors
            st.subheader("Key Risk Factors Identified")
            
            factors = []
            if glucose > 125:
                factors.append(f"**Elevated glucose level** ({glucose} mg/dL - {'Prediabetic' if glucose <= 200 else 'Diabetic'} range)")
            if bmi >= 30:
                factors.append(f"**High BMI** ({bmi}) - Obese category")
            elif bmi >= 25:
                factors.append(f"**Elevated BMI** ({bmi}) - Overweight category")
            if dpf_score > 1.0:
                factors.append(f"**Strong family history** (genetic risk score: {dpf_score:.2f})")
            if age > 45:
                factors.append(f"**Age** ({age} years - increased risk category)")
            if pregnancies >= 3:
                factors.append(f"**Multiple pregnancies** ({pregnancies})")
            if blood_pressure >= 130:
                factors.append(f"**Elevated blood pressure** ({blood_pressure} mmHg)")
            
            if factors:
                for factor in factors:
                    st.write(f"• {factor}")
            else:
                st.info("• No major risk factors identified from your inputs")
            
            # Disclaimer
            st.warning("""
            **Important Disclaimer:** This assessment is based on statistical models and should not replace 
            professional medical advice, diagnosis, or treatment. Always seek the advice of your physician 
            or other qualified health provider with any questions you may have regarding a medical condition.
            """)
            
        except Exception as e:
            st.error(f"An error occurred during assessment: {str(e)}")
            st.info("Please make sure all required model files are in the same directory.")

with tab3:
    st.header("About Diabetes")
    
    st.markdown("""
    ### What is Diabetes?
    
    Diabetes is a chronic condition that affects how your body turns food into energy. 
    There are two main types:
    
    - **Type 1 Diabetes**: An autoimmune condition where the body doesn't produce insulin
    - **Type 2 Diabetes**: A condition where the body doesn't use insulin properly (more common)
    - **Gestational Diabetes**: Develops during pregnancy and usually resolves after childbirth
    
    ### Key Risk Factors:
    
    - **High glucose levels** (≥126 mg/dL when fasting)
    - **Obesity or high BMI** (BMI ≥30 significantly increases risk)
    - **Family history** of diabetes (immediate relatives)
    - **High blood pressure** (≥140/90 mmHg)
    - **Age** (risk increases significantly after 45 years)
    - **Gestational diabetes** during pregnancy
    - **Polycystic ovary syndrome (PCOS)**
    - **Physical inactivity**
    
    ### Prevention Strategies:
    
    1. **Maintain a healthy weight**
    2. **Exercise regularly** (at least 150 minutes per week)
    3. **Eat a balanced diet** rich in fiber and low in processed foods
    4. **Limit sugar-sweetened beverages**
    5. **Avoid tobacco products**
    6. **Limit alcohol consumption**
    7. **Manage stress levels**
    8. **Get regular health check-ups**
    
    ### When to See a Doctor:
    
    Consult a healthcare professional if you experience:
    
    - Increased thirst and frequent urination
    - Extreme hunger
    - Unexplained weight loss
    - Fatigue
    - Blurred vision
    - Slow-healing sores
    - Frequent infections
    
    Early detection and management can prevent or delay complications associated with diabetes.
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Model: Optimized Random Forest | **For educational purposes only**")

# Add a sidebar with additional information
with st.sidebar:
    st.header("About This Tool")
    st.markdown("""
    This diabetes risk assessment tool uses a machine learning model trained on the Pima Indians Diabetes Dataset.
    
    **How it works:**
    1. Enter your health information
    2. The model analyzes 22 different health factors
    3. Get your personalized risk assessment
    4. Receive evidence-based recommendations
    
    **Note:** This tool provides a statistical risk assessment, not a medical diagnosis.
    
    **Model Performance:**
    - Accuracy: ~80%
    - AUC: ~0.87
    - Precision: ~0.76
    - Recall: ~0.70
    
    Always consult healthcare professionals for medical advice.
    """)
    
    if st.checkbox("Show technical details"):
        st.write("Selected features:", selected_features)
        st.write("Input data shape:", input_df_selected.shape)
