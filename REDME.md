# 🧠 Customer Churn Prediction

**Author:** [Sorrow Grajales](https://github.com/SfromtheAbyss)  
**Date:** October 2025  
**Technologies:**  Python, Pandas, Scikit-learn, XGBoost, SHAP, Streamlit
---

## 🚀 Description

This project performs customer churn prediction using Machine Learning techniques.
The goal is to identify customers at high risk of leaving the service, enabling the company to take proactive measures to improve retention and loyalty.

A complete Data Science workflow is applied, from data exploration to model deployment, including interpretation using SHAP values to understand the factors influencing churn.

---

## 🧩 Repository Structure

```
customer-churn-prediction/
├── data/
│   ├── raw/                      # original data (do not upload if large)
│   │   └── telco_customer_churn.csv
│   └── processed/
├── notebooks/
│   └── 01_eda_modeling_churn.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py                   # data loading and splitting
│   ├── preprocess.py             # feature pipelines
│   ├── models.py                 # training and evaluation
│   └── explain.py                # SHAP functions
├── app/
│   └── streamlit_app.py          # interactive demo
├── models/
│   └── best_model.joblib
├── reports/
│   └── figures/
├── requirements.txt
├── README.md
└── .gitignore
```
---

## 📊 Key Results

| Model               | ROC-AUC (val) |
| ------------------- | ------------- |
| Logistic Regression | **0.8578**    |
| Random Forest       | 0.8560        |
| XGBoost             | 0.8507        |

📈 **Metrics of the selected model (final evaluation):**

| Metric    | Value  |
| --------- | ------ |
| ROC-AUC   | 0.8407 |
| Precision | 0.6763 |
| Recall    | 0.4358 |
| F1-Score  | 0.5301 |

**Confusion Matrix:**

```
[[957, 78],
[211, 163]]
```
---

## ⚙️ Technologies Used

- Python 3.11  
- pandas, numpy, matplotlib, seaborn  
- scikit-learn  
- xgboost  
- shap  
- joblib  

---

## 🧠 Workflow Pipeline

1. **Load data** using `src/data.py`
2. **Preprocess data** via `build_preprocessor()`  
   - Scaling numerical variables
   - OneHotEncoding / Target Encoding depending on cardinality
3. **Train models** with `build_pipeline()`
4. **Evaluate and compare** models
5. **Interpret results** using SHAP
6. **Save final model** (`best_model.joblib`)

---

## 📦 Installation and Usage

### 1️⃣ Clone the repository
```
git clone git@github.com:SfromtheAbyss/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2️⃣ Create virtual environment
```
conda create -n churn_env python=3.11
conda activate churn_env
```

### 3️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 4️⃣ Run the main notebook
```
jupyter notebook notebooks/01_eda_modeling_churn.ipynb
```

### 5️⃣ Load and use the model
```
import joblib
model = joblib.load('models/best_model.joblib')
pred = model.predict_proba(nuevo_cliente)[:,1]
```

---

## 🧭 Next Steps

Implement an interactive Streamlit dashboard.

Automate model training and tracking with MLflow.

Deploy the model via a REST API using FastAPI.

---

## 👤 Author

📌 Sorrow Grajales
📍 Barcelona, España
🔗 [LinkedIn](https://linkedin.com/in/sforsorrow)
💻 [GitHub](https://github.com/SfromtheAbyss)

---

## ✨ “Predicting the future is hard, but with data, we can take one step closer.”