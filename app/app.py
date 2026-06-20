import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

# ----------------------------------------------------------------------
# 1️⃣ Load AWS credentials from Streamlit Secrets (you will fill these in the UI)
# ----------------------------------------------------------------------
# The secrets should be defined in the Streamlit Cloud UI under Settings → Secrets:
# [aws]
# access_key_id = "YOUR_ACCESS_KEY_ID"
# secret_access_key = "YOUR_SECRET_ACCESS_KEY"
# bucket_name = "your-bucket-name"

aws_cfg = st.secrets.get("aws", {})
AWS_ACCESS_KEY_ID = aws_cfg.get("access_key_id")
AWS_SECRET_ACCESS_KEY = aws_cfg.get("secret_access_key")
BUCKET_NAME = aws_cfg.get("bucket_name")

if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BUCKET_NAME]):
    st.error("AWS credentials not configured in Streamlit Secrets. Please add them before running the app.")
    st.stop()

# ----------------------------------------------------------------------
# 2️⃣ Helper: download a file from S3 only if it does not already exist locally
# ----------------------------------------------------------------------
def download_from_s3(s3_key: str, local_path: str):
    """Download an object from the configured S3 bucket to ``local_path``.
    The function is idempotent – it skips the download if the file already exists.
    """
    if os.path.exists(local_path):
        return
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
        s3_client.download_file(BUCKET_NAME, s3_key, local_path)
        st.info(f"Downloaded {s3_key} from bucket {BUCKET_NAME}.")
    except (NoCredentialsError, ClientError) as e:
        st.error(f"Failed to download {s3_key} from S3: {e}")
        st.stop()

# ----------------------------------------------------------------------
# 3️⃣ Ensure the ``model`` directory exists and fetch the artefacts
# ----------------------------------------------------------------------
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")

download_from_s3("churn_model.pkl", MODEL_PATH)
download_from_s3("scaler.pkl", SCALER_PATH)
download_from_s3("feature_columns.pkl", FEATURES_PATH)

# ----------------------------------------------------------------------
# 4️⃣ Load the model, scaler, and feature column order
# ----------------------------------------------------------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
FEATURE_COLUMNS = joblib.load(FEATURES_PATH)

# ----------------------------------------------------------------------
# 5️⃣ Pre‑processing helpers – must mirror notebook logic exactly
# ----------------------------------------------------------------------
BINARY_MAP = {"Yes": 1, "No": 0, "No internet service": 0, "No phone service": 0}
BINARY_COLS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
]
ONEHOT_COLS = ["Contract", "InternetService", "PaymentMethod"]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def preprocess_input(raw_dict: dict) -> pd.DataFrame:
    """Convert raw user input (dict) into the exact feature matrix used for training.
    The steps are:
    1️⃣ Map binary Yes/No values (including "No phone service" / "No internet service")
    2️⃣ One‑hot encode the three nominal columns, filling missing dummy columns with zeros
    3️⃣ Scale numeric columns with the fitted StandardScaler
    4️⃣ Re‑order columns to ``FEATURE_COLUMNS``
    """
    df = pd.DataFrame([raw_dict])

    # Binary mapping
    for col in BINARY_COLS:
        df[col] = df[col].replace(BINARY_MAP).astype(int)

    # One‑hot encoding – ensure we have *all* columns that existed during training
    df_onehot = pd.get_dummies(df[ONEHOT_COLS], drop_first=False)
    # Add any missing one‑hot columns (set to 0)
    for col in FEATURE_COLUMNS:
        if col not in df_onehot.columns and any(col.startswith(p + "_") for p in ONEHOT_COLS):
            df_onehot[col] = 0
    df = pd.concat([df.drop(columns=ONEHOT_COLS), df_onehot], axis=1)

    # Scale numeric features
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    # Final column ordering
    df = df[FEATURE_COLUMNS]
    return df

# ----------------------------------------------------------------------
# 6️⃣ Streamlit UI
# ----------------------------------------------------------------------
st.title('📞 Telco Customer Churn Prediction')
st.markdown('Enter the customer details below and press **Predict** to obtain a churn probability.')

with st.form(key='churn_form'):
    gender = st.selectbox('Gender', options=['Female', 'Male'])
    senior = st.selectbox('Senior Citizen', options=['No', 'Yes'])
    partner = st.selectbox('Partner', options=['No', 'Yes'])
    dependents = st.selectbox('Dependents', options=['No', 'Yes'])
    tenure = st.number_input('Tenure (months)', min_value=0, max_value=72, value=0)
    phone_service = st.selectbox('Phone Service', options=['No', 'Yes'])
    multiple_lines = st.selectbox('Multiple Lines', options=['No', 'Yes', 'No phone service'])
    internet_service = st.selectbox('Internet Service', options=['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('Online Security', options=['No', 'Yes', 'No internet service'])
    online_backup = st.selectbox('Online Backup', options=['No', 'Yes', 'No internet service'])
    device_protection = st.selectbox('Device Protection', options=['No', 'Yes', 'No internet service'])
    tech_support = st.selectbox('Tech Support', options=['No', 'Yes', 'No internet service'])
    streaming_tv = st.selectbox('Streaming TV', options=['No', 'Yes', 'No internet service'])
    streaming_movies = st.selectbox('Streaming Movies', options=['No', 'Yes', 'No internet service'])
    contract = st.selectbox('Contract', options=['Month-to-month', 'One year', 'Two year'])
    paperless = st.selectbox('Paperless Billing', options=['No', 'Yes'])
    payment_method = st.selectbox('Payment Method', options=[
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ])
    monthly_charges = st.number_input('Monthly Charges', min_value=0.0, max_value=200.0, value=0.0, step=0.1)
    total_charges = st.number_input('Total Charges', min_value=0.0, max_value=10000.0, value=0.0, step=0.1)

    submit = st.form_submit_button('Predict')

if submit:
    raw = {
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
    }
    try:
        X_input = preprocess_input(raw)
        prob = model.predict_proba(X_input)[:, 1][0]
        pred = 'Churn' if prob >= 0.5 else 'No Churn'
        # Risk level thresholds
        if prob < 0.30:
            risk, color = 'Low', 'green'
        elif prob < 0.60:
            risk, color = 'Medium', 'orange'
        else:
            risk, color = 'High', 'red'
        st.success(f'**Prediction:** {pred}')
        st.metric(label='Churn Probability (%)', value=f"{prob * 100:.1f}%", delta=risk)
        st.markdown(f"<span style='color:{color};font-weight:bold'>Risk Level: {risk}</span>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f'Error during prediction: {e}')

st.sidebar.title('About')
st.sidebar.info('This app uses a Random Forest model trained on the IBM Telco Customer Churn dataset.\n\nInterpretation of predictions should consider that the model captures correlations, not causation.')
