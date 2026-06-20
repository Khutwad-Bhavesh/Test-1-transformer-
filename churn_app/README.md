# Telecom Customer Churn Prediction

A Streamlit app that predicts whether a telecom customer will churn, based on
account usage and billing behavior.

## 📂 Project Structure
```
churn_app/
├─ app/
│  └─ app.py              # Streamlit inference app
├─ data/
│  └─ telecom_churn.csv   # Training data
├─ model/                 # Trained artifacts (committed to the repo)
│  ├─ churn_model.pkl
│  ├─ scaler.pkl
│  └─ feature_columns.pkl
├─ train_model.py         # Regenerates the artifacts in ./model/
├─ requirements.txt
└─ README.md
```

## 📊 About the model
- **Algorithm**: Random Forest (300 trees, max depth 10), `class_weight="balanced"`
  to account for the ~15% churn rate in the data.
- **Features**: account length, contract renewal status, data plan, data usage,
  customer service calls, day minutes/calls, monthly charge, overage fee, and
  roaming minutes.
- **Test performance**: ROC-AUC ≈ 0.85, recall ≈ 0.75 on churners.

## 🛠️ Run locally
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# (Optional) retrain the model — artifacts are already included in ./model/
python train_model.py

streamlit run app/app.py
```
Open the URL shown (typically http://localhost:8501).

## 🚀 Deploy to Streamlit Community Cloud
1. Commit and push this whole folder (including `model/*.pkl`) to your GitHub repo.
   The model files are ~3 MB total, well within GitHub's limits — no Git LFS needed.
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **"New app"**, select your repo and branch.
4. Set **Main file path** to `app/app.py`.
5. Click **Deploy**. Streamlit Cloud installs `requirements.txt` automatically
   and the app loads the model artifacts straight from the repo — no secrets
   or external services required.
6. Once live, you'll get a shareable `*.streamlit.app` URL.

> If you ever retrain the model (e.g. with new data), just re-run
> `python train_model.py`, commit the updated files in `model/`, and push —
> Streamlit Cloud will redeploy automatically.
