import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Telecom Churn Prediction", page_icon="📞")

# ----------------------------------------------------------------------
# Load model artifacts (bundled in the repo under ./model/)
# ----------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, scaler, feature_columns


try:
    model, scaler, FEATURE_COLUMNS = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model artifacts not found in ./model/. "
        "Run `python train_model.py` first to generate churn_model.pkl, "
        "scaler.pkl, and feature_columns.pkl."
    )
    st.stop()

BINARY_COLS = ["ContractRenewal", "DataPlan"]
NUMERIC_COLS = [
    "AccountWeeks", "DataUsage", "CustServCalls", "DayMins",
    "DayCalls", "MonthlyCharge", "OverageFee", "RoamMins",
]


def preprocess_input(raw_dict: dict) -> pd.DataFrame:
    """Convert raw user input into the exact feature matrix used for training."""
    df = pd.DataFrame([raw_dict])
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
    return df[FEATURE_COLUMNS]


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("📞 Telecom Customer Churn Prediction")
st.markdown(
    "Enter the customer's account details below and press **Predict** "
    "to estimate their probability of churning."
)

with st.form(key="churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        account_weeks = st.number_input(
            "Account Length (weeks)", min_value=0, max_value=260, value=100
        )
        contract_renewal = st.selectbox(
            "Recently Renewed Contract?", options=["Yes", "No"]
        )
        data_plan = st.selectbox("Has Data Plan?", options=["Yes", "No"])
        data_usage = st.number_input(
            "Data Usage (GB)", min_value=0.0, max_value=10.0, value=0.8, step=0.1
        )
        cust_serv_calls = st.number_input(
            "Customer Service Calls", min_value=0, max_value=15, value=1
        )

    with col2:
        day_mins = st.number_input(
            "Day Minutes Used", min_value=0.0, max_value=400.0, value=180.0, step=1.0
        )
        day_calls = st.number_input(
            "Day Calls Made", min_value=0, max_value=200, value=100
        )
        monthly_charge = st.number_input(
            "Monthly Charge ($)", min_value=0.0, max_value=200.0, value=56.0, step=1.0
        )
        overage_fee = st.number_input(
            "Overage Fee ($)", min_value=0.0, max_value=30.0, value=10.0, step=0.5
        )
        roam_mins = st.number_input(
            "Roaming Minutes", min_value=0.0, max_value=30.0, value=10.0, step=0.5
        )

    submit = st.form_submit_button("Predict")

if submit:
    raw = {
        "AccountWeeks": account_weeks,
        "ContractRenewal": 1 if contract_renewal == "Yes" else 0,
        "DataPlan": 1 if data_plan == "Yes" else 0,
        "DataUsage": data_usage,
        "CustServCalls": cust_serv_calls,
        "DayMins": day_mins,
        "DayCalls": day_calls,
        "MonthlyCharge": monthly_charge,
        "OverageFee": overage_fee,
        "RoamMins": roam_mins,
    }
    try:
        X_input = preprocess_input(raw)
        prob = model.predict_proba(X_input)[:, 1][0]
        pred = "Churn" if prob >= 0.5 else "No Churn"

        if prob < 0.30:
            risk, color = "Low", "green"
        elif prob < 0.60:
            risk, color = "Medium", "orange"
        else:
            risk, color = "High", "red"

        st.success(f"**Prediction:** {pred}")
        st.metric(label="Churn Probability", value=f"{prob * 100:.1f}%", delta=risk)
        st.markdown(
            f"<span style='color:{color};font-weight:bold'>Risk Level: {risk}</span>",
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.sidebar.title("About")
st.sidebar.info(
    "This app uses a Random Forest model trained on a telecom customer "
    "churn dataset (account usage and billing features).\n\n"
    "Predictions reflect correlations learned by the model, not causation."
)
