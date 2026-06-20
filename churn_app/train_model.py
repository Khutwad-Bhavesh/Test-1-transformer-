"""
Train a churn-prediction model on the telecom_churn dataset.

Run:
    python train_model.py

Produces (in ./model/):
    churn_model.pkl       - trained RandomForestClassifier
    scaler.pkl            - StandardScaler fit on the numeric features
    feature_columns.pkl   - exact column order expected at inference time
"""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

DATA_PATH = os.path.join("data", "telecom_churn.csv")

# Columns that are already binary (0/1) - no encoding needed
BINARY_COLS = ["ContractRenewal", "DataPlan"]

# Continuous numeric columns - will be standardized
NUMERIC_COLS = [
    "AccountWeeks", "DataUsage", "CustServCalls", "DayMins",
    "DayCalls", "MonthlyCharge", "OverageFee", "RoamMins",
]

FEATURE_COLUMNS = BINARY_COLS + NUMERIC_COLS
TARGET_COL = "Churn"


def main():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scale only the continuous numeric columns; binary flags stay as-is
    scaler = StandardScaler()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Quick evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print("Test set performance:")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"  Recall   : {recall_score(y_test, y_pred):.3f}")
    print(f"  F1       : {f1_score(y_test, y_pred):.3f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, y_proba):.3f}")

    # Feature importance (sanity check)
    importances = sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print("\nTop features:")
    for name, imp in importances[:5]:
        print(f"  {name}: {imp:.3f}")

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, os.path.join("model", "churn_model.pkl"))
    joblib.dump(scaler, os.path.join("model", "scaler.pkl"))
    joblib.dump(FEATURE_COLUMNS, os.path.join("model", "feature_columns.pkl"))
    print("\nArtifacts saved in ./model/")


if __name__ == "__main__":
    main()
