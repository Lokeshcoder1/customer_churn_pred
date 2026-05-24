import streamlit as st
import requests

API_DEFAULT = "http://localhost:8000/predict"


def main():
    st.set_page_config(page_title="Churn Prediction")
    st.title("Customer Churn Prediction")

    st.sidebar.header("API Settings")
    api_url = st.sidebar.text_input("Prediction endpoint", API_DEFAULT)

    with st.form("input_form"):
        tenure = st.number_input("Tenure (months)", min_value=1.0, value=24.0)
        monthly_charges = st.number_input("Monthly charges", min_value=0.0, value=65.5)
        total_charges = st.number_input("Total charges", min_value=0.0, value=1570.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
        senior_citizen = st.selectbox("Senior citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        phone_service = st.selectbox("Phone service", ["Yes", "No"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming movies", ["Yes", "No", "No internet service"])
        paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment method", ["Bank transfer", "Credit card", "Electronic check", "Mailed check"])

        submit = st.form_submit_button("Predict")

    if submit:
        payload = {
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract": contract,
            "internet_service": internet_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "gender": gender,
            "phone_service": phone_service,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
        }

        try:
            resp = requests.post(api_url, json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                st.success(f"Churn probability: {data.get('churn_probability'):.2f}")
                st.write("Prediction:", data.get('churn_prediction'))
                st.write("Confidence:", data.get('confidence'))
            else:
                st.error(f"API error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")


if __name__ == "__main__":
    main()
