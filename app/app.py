import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ------------------------------------------------------------
# Load model artifacts
# ------------------------------------------------------------
MODEL_PATH = os.path.join('model', 'churn_model.pkl')
SCALER_PATH = os.path.join('model', 'scaler.pkl')
FEATURES_PATH = os.path.join('model', 'feature_columns.pkl')

@st.cache_resource

def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    return model, scaler, feature_cols

model, scaler, FEATURE_COLUMNS = load_artifacts()

# ------------------------------------------------------------
# Helper: preprocessing – must match notebook exactly
# ------------------------------------------------------------
BINARY_MAP = {'Yes': 1, 'No': 0, 'No internet service': 0, 'No phone service': 0}

BINARY_COLS = [
    'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'PaperlessBilling'
]

ONEHOT_COLS = ['Contract', 'InternetService', 'PaymentMethod']

NUMERIC_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

def preprocess_input(raw_dict):
    """Convert raw user input (as dict) into the model feature vector.
    The steps mirror the notebook preprocessing.
    """
    df = pd.DataFrame([raw_dict])

    # Map binary Yes/No columns (including service‑no cases)
    for col in BINARY_COLS:
        df[col] = df[col].replace(BINARY_MAP).astype(int)

    # One‑hot encode nominal columns – ensure all columns from training are present
    df_onehot = pd.get_dummies(df[ONEHOT_COLS], drop_first=False)
    # Add missing one‑hot columns with zeros
    for col in FEATURE_COLUMNS:
        if col not in df_onehot.columns and col not in df.columns:
            # Only add columns that originated from one‑hot encoding
            if any(col.startswith(prefix + '_') for prefix in ONEHOT_COLS):
                df_onehot[col] = 0
    df = pd.concat([df.drop(columns=ONEHOT_COLS), df_onehot], axis=1)

    # Scale numeric columns (StandardScaler fitted on training data)
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    # Re‑order columns to exact training order
    df = df[FEATURE_COLUMNS]
    return df

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.title('📞 Telco Customer Churn Prediction')
st.markdown('Enter the customer details below and press **Predict** to see the churn probability.')

# ---- Input widgets ----
with st.form(key='churn_form'):
    # Categorical inputs (selectbox)
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
    # Build raw dictionary matching original column names (except customerID)
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
        'TotalCharges': total_charges
    }

    try:
        X_input = preprocess_input(raw)
        prob = model.predict_proba(X_input)[:, 1][0]
        pred = 'Churn' if prob >= 0.5 else 'No Churn'

        # Risk level thresholds
        if prob < 0.30:
            risk = 'Low'
            color = 'green'
        elif prob < 0.60:
            risk = 'Medium'
            color = 'orange'
        else:
            risk = 'High'
            color = 'red'

        st.success(f'**Prediction:** {pred}')
        st.metric(label='Churn Probability (%)', value=f"{prob * 100:.1f}%", delta=risk)
        st.markdown(f"<span style='color:{color};font-weight:bold'>Risk Level: {risk}</span>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f'Error during prediction: {e}')

st.sidebar.title('About')
st.sidebar.info('This app uses a Random Forest model trained on the IBM Telco Customer Churn dataset.\n\n'\
                'Interpretation of predictions should consider that the model captures correlations, not causation.')
