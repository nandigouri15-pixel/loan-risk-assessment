import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Page Configuration Setup
st.set_page_config(
    page_title="Credit Risk Analyzer",
    page_icon="🏦",
    layout="centered"
)

# 2. Design UI Header elements
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🏦 Smart Loan Risk Assessment Portal</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #6B7280;'>Production-Grade Machine Learning Evaluation Framework</h4>", unsafe_allow_html=True)
st.write("---")

# 3. Automatic Model Training Pipeline 
@st.cache_resource
def train_and_cache_model():
    # Load dataset natively 
    df = pd.read_csv("loan_data.csv")
    
    # Preprocess and Encode
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    X = pd.get_dummies(X, drop_first=True)
    
    # Train the Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, list(X.columns)

# Safely call training block
try:
    model, trained_features = train_and_cache_model()
    st.sidebar.success("🤖 Core ML Model Trained & Loaded Live!")
except Exception as e:
    st.error(f"⚠️ Dataset linking error. Make sure 'loan_data.csv' is uploaded to your GitHub repository root folder. Details: {e}")
    st.stop()

# 4. Technical Summary in the Sidebar Layout (Interviewer Cheat Sheet)
with st.sidebar:
    st.write("---")
    st.markdown("### 🛠️ System Architecture")
    st.markdown("""
    - **Classifier Engine:** Random Forest Ensemble Model
    - **Features Processing:** Categorical One-Hot Encoding
    - **Workflow Integration:** Live Dynamic Cache
    - **Data Pipeline Size:** 1,000+ Native Transactions
    - **Developer Profile:** [nandigouri15-pixel](https://github.com)
    """)

# 5. Form Interface Structure for Applicant Metrics
st.subheader("📋 Enter Applicant Demographics & Financial Details")

col1, col2 = st.columns(2)

with col1:
    person_age = st.slider("Applicant Age", 18, 100, 28)
    person_income = st.number_input("Annual Income ($)", min_value=0, value=55000, step=1000)
    person_gender = st.selectbox("Gender", ["male", "female"])
    person_education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "Doctorate", "Associate"])

with col2:
    loan_amnt = st.number_input("Requested Loan Amount ($)", min_value=0, value=12000, step=500)
    credit_score = st.slider("Credit Score", 300, 850, 680)
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
    loan_intent = st.selectbox("Loan Purpose Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])

# 6. Process Input Data on Button Click
st.write("")
if st.button("🚀 Analyze Credit & Predict Loan Risk", use_container_width=True):
    
    st.write("---")
    st.subheader("📊 Analytical Decision Summary")
    
    # A. Business Logic Guardrail Check
    if loan_amnt > (person_income * 3):
        st.warning("⚠️ **Financial Guardrail Notice:** The requested loan amount exceeds 300% of the applicant's annual income. This profile carries structural high-leverage risk.")
        st.write("")
    
    # B. Construct a raw dictionary mimicking your original dataset structure
    input_data = {
        'person_age': person_age,
        'person_income': person_income,
        'loan_amnt': loan_amnt,
        'credit_score': credit_score,
        'person_gender': person_gender,
        'person_education': person_education,
        'person_home_ownership': person_home_ownership,
        'loan_intent': loan_intent
    }
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Apply One-Hot Encoding to the input row
    input_encoded = pd.get_dummies(input_df)
    
    # Reindex columns to match trained template schema layout
    input_final = input_encoded.reindex(columns=trained_features, fill_value=0)
    
    # Compute probabilities and final predictions
    prediction = model.predict(input_final)
    probabilities = model.predict_proba(input_final)
    
    # Get the specific probability for risk class '1' (Default Risk)
    # probabilities[0][1] gives the chance of default (class 1)
    default_prob_percentage = float(probabilities[0][1]) * 100
    
    # C. Visual Risk Gauge Bar & Response Handling
    st.write(f"**Calculated Algorithmic Default Risk Metric:**")
    st.progress(default_prob_percentage / 100)
    
    if prediction == 1 or default_prob_percentage > 50:
        st.error(f"❌ **Loan Request Flagged (High Risk)**")
        st.write(f"The algorithmic validation pipeline flags this profile as a potential credit risk. Calculated Default Probability: **{default_prob_percentage:.2f}%**")
    else:
        st.success(f"✅ **Loan Request Approved (Low Risk)**")
        st.write(f"The algorithmic validation pipeline passes this profile safely. Calculated Credit Default Risk: **{default_prob_percentage:.2f}%**")
