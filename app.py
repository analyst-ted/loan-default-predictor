
import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd

# --- 1. SET UP THE PAGE ---
st.set_page_config(page_title="Startup Loan Approver", page_icon="💸")
st.title("💸 AI Risk Assessment Portal")
st.write("Enter the applicant's financial details below to predict default risk.")

# --- 2. LOAD THE ASSETS ---
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('loan_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()
# The magic blueprint: exact 71 columns the model expects
expected_columns = scaler.feature_names_in_ 

# --- 3. BUILD THE USER INTERFACE ---
st.header("Applicant Profile")

col1, col2 = st.columns(2)

with col1:
    loan_amnt = st.number_input("Requested Loan Amount (Rs.)", min_value=1000, max_value=40000, value=10000)
    int_rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=30.0, value=12.5)
    annual_inc = st.number_input("Annual Income (Rs.)", min_value=10000, max_value=250000, value=65000)
    emp_length = st.slider("Employment Length (Years)", min_value=0, max_value=10, value=5)

with col2:
    # We add a few categorical dropdowns to make the app realistic!
    home_ownership = st.selectbox("Home Ownership", ['RENT', 'MORTGAGE', 'OWN'])
    purpose = st.selectbox("Loan Purpose", ['debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase', 'small_business'])
    sub_grade = st.selectbox("Lending Club Sub-Grade", ['A1', 'B2', 'C3', 'D4', 'E5', 'F1', 'G5'])

# --- 4. THE PREDICTION ENGINE ---
st.markdown("---")
if st.button("Run AI Prediction", type="primary"):
    
    # 1. Create a blank dictionary with all 71 columns set to 0.0
    input_dict = {col: 0.0 for col in expected_columns}
    
    # 2. Fill in the numerical values the user typed
    # (Make sure these names exactly match your original CSV column names!)
    if 'loan_amnt' in input_dict: input_dict['loan_amnt'] = loan_amnt
    if 'int_rate' in input_dict: input_dict['int_rate'] = int_rate
    if 'annual_inc' in input_dict: input_dict['annual_inc'] = annual_inc
    if 'emp_length' in input_dict: input_dict['emp_length'] = emp_length
    
    # 3. Flip the switch (set to 1.0) for the dropdown categories they selected
    # Example: If they chose RENT, we look for 'home_ownership_RENT'
    if f'home_ownership_{home_ownership}' in input_dict:
        input_dict[f'home_ownership_{home_ownership}'] = 1.0
        
    if f'purpose_{purpose}' in input_dict:
        input_dict[f'purpose_{purpose}'] = 1.0
        
    if f'sub_grade_{sub_grade}' in input_dict:
        input_dict[f'sub_grade_{sub_grade}'] = 1.0
        
    # 4. Convert the dictionary into a DataFrame with exactly 1 row
    input_df = pd.DataFrame([input_dict])
    
    # 5. Scale it and Predict!
    scaled_input = scaler.transform(input_df)
    prediction_prob = model.predict(scaled_input)[0][0] # Get the raw probability
    
    # 6. Apply the threshold you discovered earlier!
    threshold = 0.60
    
    st.subheader("AI Decision:")
    if prediction_prob >= threshold:
        st.success(f"✅ APPROVED. The AI is {prediction_prob*100:.1f}% confident this loan will be fully paid.")
    else:
        st.error(f"❌ REJECTED. The AI is only {prediction_prob*100:.1f}% confident. High risk of default.")