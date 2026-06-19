import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load data
DATA_PATH = os.path.join('data', 'Telco-Customer-Churn.csv')
df = pd.read_csv(DATA_PATH)

# Clean TotalCharges (convert to numeric, fill NaN with 0 for tenure==0)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(0, inplace=True)

# Preprocessing
df_processed = df.copy()
# Drop identifier
df_processed.drop('customerID', axis=1, inplace=True)
# Collapse service‑no values
service_no_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in service_no_cols:
    df_processed[col] = df_processed[col].replace({
        'No internet service': 'No',
        'No phone service': 'No'
    })
# Binary mapping (Yes/No -> 1/0)
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn'] + service_no_cols
for col in binary_cols:
    df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0})
# One‑hot encode nominal columns
nominal_cols = ['Contract', 'InternetService', 'PaymentMethod']
df_processed = pd.get_dummies(df_processed, columns=nominal_cols, drop_first=False)
# Separate target
y = df_processed['Churn']
X = df_processed.drop('Churn', axis=1)
# Scale numeric features (only needed for Logistic Regression, harmless for trees)
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
# Train‑test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# Train models (class_weight='balanced')
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
log_reg.fit(X_train, y_train)

dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=30,
                            class_weight='balanced', random_state=42)

dt.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=10,
                            class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Save final model (Random Forest) and preprocessing objects
os.makedirs('model', exist_ok=True)
joblib.dump(rf, os.path.join('model', 'churn_model.pkl'))
joblib.dump(scaler, os.path.join('model', 'scaler.pkl'))
joblib.dump(list(X_train.columns), os.path.join('model', 'feature_columns.pkl'))

print('Training complete. Artifacts saved in ./model/')
