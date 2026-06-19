# Customer Churn Prediction System

## 📌 Problem Statement
A telecom company wants to identify customers who are likely to stop using its services so that preventive actions can be taken to improve customer retention.

## 🎯 Objective
Build a Machine Learning model that predicts whether a customer will churn and create a Streamlit application for prediction.

## 📊 Dataset
IBM Telco Customer Churn Dataset (Kaggle)

## 🛠 Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Jupyter Notebook

## 🔍 Data Preprocessing
- Removed unnecessary columns
- Converted Total Charges to numeric values
- Handled missing values
- Encoded categorical variables using LabelEncoder

## 🤖 Machine Learning Models Used
1. Logistic Regression
2. Decision Tree
3. Random Forest

## 📈 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score

## 🏆 Final Model Selected
Random Forest Classifier

### Model Performance

| Model | Accuracy |
|---------|---------|
| Logistic Regression | 79.13% |
| Decision Tree | 72.75% |
| Random Forest | 80.13% |

## ⭐ Important Features Affecting Churn
- Monthly Charges
- Tenure Months
- Contract Type
- Total Charges
- CLTV

## 💡 Business Recommendations
- Customers with high monthly charges are more likely to churn.
- Customers with low tenure are at higher risk of leaving.
- Long-term contracts improve customer retention.
- Providing Tech Support and Online Security can reduce churn.
- High-risk customers should be targeted with special offers.

## 🚀 Streamlit Application
The application allows users to:
- Enter customer details
- Predict customer churn
- View churn probability
- Determine whether the customer is likely to stay or leave

## 📂 Project Structure

```
Customer-Churn-Prediction/
│
├── main.py
├── app.py
├── model.pkl
├── Customer_Churn.ipynb
├── Telco_customer_churn.xlsx
├── README.md
├── requirements.txt
```

## ▶️ How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit App

```bash
streamlit run app.py
```

## 📌 Author
Omkar Khandekar