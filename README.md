# Customer Churn Prediction Project

## 📌 Key Decisions
| Decision | Rationale |
|---|---|
| **Class imbalance handling** | Used `class_weight='balanced'` for all models instead of oversampling – keeps original data distribution while penalising mis‑classification of the minority **Churn** class. |
| **Metric choice** | Prioritised **Recall** and **ROC‑AUC** over raw accuracy because false negatives (missed churners) are far more costly than false positives. |
| **Model selection** | Chose **Random Forest** (300 trees, `max_depth=10`) – best F1/ROC‑AUC and captures non‑linear feature interactions without heavy tuning. |
| **Missing `TotalCharges`** | Converted to numeric; rows with `NaN` correspond to customers with `tenure=0` (new sign‑ups). Filled with `0` rather than dropping to retain early‑stage information. |
| **Encoding strategy** | One‑hot encoded nominal columns (`Contract`, `InternetService`, `PaymentMethod`) – avoids imposing ordinal relationships that label‑encoding would create. |

## 🛠️ Setup & Run
### Prerequisites (Linux – Arch example)
```bash
# System packages (numpy, pandas, scikit‑learn, etc.)
sudo pacman -Syu python python-pip python-venv
```
### Prerequisites (Windows)
1. Install **Python 3.10+** from the official installer (https://www.python.org/downloads/windows/). During installation, check *Add Python to PATH*.
2. Open **PowerShell** (or Command Prompt) and verify installation:
```powershell
python --version
pip --version
```
3. (Optional) Install **Git** if you wish to clone the repo.

### Create virtual environment & install Streamlit (common for both OSes)
```bash
# Linux/macOS
cd /home/aizen-sosuke/Projects/churn_project
python -m venv venv --system-site-packages   # re‑use system libs for speed
source venv/bin/activate
# Windows PowerShell
cd C:\Path\To\churn_project
python -m venv venv --system-site-packages
venv\Scripts\Activate.ps1
# Common steps (both platforms)
pip install --upgrade pip
pip install streamlit joblib jupyterlab
```
### Run the notebook (exploratory analysis & model training)
```bash
# Linux/macOS
jupyter notebook notebook/Customer_Churn_Prediction.ipynb
# Windows PowerShell
jupyter notebook notebook\Customer_Churn_Prediction.ipynb
```
* Execute all cells – the notebook will generate the trained model and three artifact files under `model/`.

### Launch the Streamlit app
```bash
# Linux/macOS
streamlit run app/app.py
# Windows PowerShell
streamlit run app\app.py
```
Open the URL shown (typically http://localhost:8501) and fill the form to obtain a churn prediction.

## 📂 Project Structure
```
churn_project/
├─ notebook/                # Jupyter notebook with analysis, training, and justification
│   └─ Customer_Churn_Prediction.ipynb
├─ model/                   # Serialized artifacts created by the notebook
│   ├─ churn_model.pkl      # Final Random Forest model
│   ├─ scaler.pkl           # StandardScaler for numeric features
│   └─ feature_columns.pkl # Column order used during training
├─ app/                     # Streamlit inference app
│   └─ app.py
└─ README.md                # This file
```

## 📋 Business Recommendations (from the notebook)
1. **Offer longer contracts** (e.g., 2‑year) to low‑tenure customers – contracts are the top protective feature.
2. **Bundle TechSupport & OnlineSecurity** with fiber‑optic plans – both appear in the top‑15 importance list.
3. **Encourage autopay** for senior customers – reduces churn among the senior segment.
4. **Review pricing** for high‑monthly‑charge customers without add‑ons – high charges correlate with churn.
5. **Targeted retention** – prioritize customers with predicted churn probability > 60 % (high risk) and 30‑60 % (medium risk) using personalized offers.

> **⚠️ Caveat**: All insights stem from correlations learned by the model; causal relationships require A/B testing or controlled experiments.
