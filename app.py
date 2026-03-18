import streamlit as st  
import joblib  
import numpy as np  
model = joblib.load('churn_model.pkl')  
scaler = joblib.load('scaler.pkl')  
st.title("?? Customer Churn Predictor") 
r = st.slider("Days since last purchase", 0, 100, 25)  
f = st.number_input("Total Orders", 1, 50, 5)  
m = st.number_input("Total Spent ($)", 10, 5000, 200)  
if st.button("Predict Risk"):  
    feat = np.array([[r, f, m, m/f]])  
    s_feat = scaler.transform(feat)  
    p = model.predict_proba(s_feat)[0][1]  
    if p > 0.7: st.error(f"High Risk: {p:.2%}")  
    elif p > 0.3: st.warning(f"Medium Risk: {p:.2%}")  
    else: st.success(f"Low Risk: {p:.2%}") 
