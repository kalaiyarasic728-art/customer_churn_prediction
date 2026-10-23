import streamlit as st
import pickle

# load model
model = pickle.load(open("models/churn_model.pkl", "rb"))

st.title("Customer Churn Prediction System")

st.write("Enter customer details below:")

# =========================
# INPUTS
# =========================
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
tenure = st.number_input("Tenure (months)", 0, 100)
charges = st.number_input("Monthly Charges", 0.0, 1000.0)

# =========================
# ENCODING (same as training)
# =========================
gender = 1 if gender == "Male" else 0

# =========================
# PREDICTION
# =========================
if st.button("Predict"):
    input_data = [[gender, senior, tenure, charges]]
    
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("❌ Customer will CHURN")
    else:
        st.success("✅ Customer will STAY")