import pickle
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

data = load_model()
model = data["model"]
encoders = data["encoders"]
columns = data["columns"]

st.title("🏦 Loan Approval Predictor")
st.write("Fill in the applicant's details and click Predict.")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender ebad", encoders["Gender"].classes_)
    married = st.selectbox("Married", encoders["Married"].classes_)
    education = st.selectbox("Education", encoders["Education"].classes_)
    self_employed = st.selectbox("Self Employed", encoders["Self_Employed"].classes_)
    property_area = st.selectbox("Property Area", encoders["Property_Area"].classes_)

with col2:
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount (thousands)", min_value=0, value=120)
    loan_term = st.number_input("Loan Term (days)", min_value=0, value=360)
    credit_history = st.selectbox("Credit History (1 = good, 0 = bad)", [1.0, 0.0])

if st.button("Predict", type="primary"):
    row = {
        "Gender ebad": encoders["Gender"].transform([gender])[0],
        "Married": encoders["Married"].transform([married])[0],
        "Education": encoders["Education"].transform([education])[0],
        "Self_Employed": encoders["Self_Employed"].transform([self_employed])[0],
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": encoders["Property_Area"].transform([property_area])[0],
    }
    X_new = pd.DataFrame([row])[columns]
    pred = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0][1]

    if pred == 1:
        st.success(f"✅ Loan likely Approved — probability {proba:.1%}")
    else:
        st.error(f"❌ Loan likely Rejected — probability of approval {proba:.1%}")
