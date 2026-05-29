# 📉 Customer Churn Prediction – End-to-End ML App

A production‑ready machine learning application that predicts telecom customer churn with **86% ROC‑AUC**. Designed for business users to proactively identify at‑risk customers and reduce revenue loss.  
**Live app:** *(your Render URL here)*

---

## 🚀 Key Highlights

- **Two models with clear trade‑offs:** Logistic Regression (explainable) vs Random Forest (higher accuracy)
- **Full preprocessing pipeline:** Handles missing values, encoding, scaling, and SMOTE imbalance correction—all packed into a single scikit‑learn Pipeline object
- **Interactive Streamlit interface:** Manual prediction, batch CSV upload, demo mode, model comparison, and performance dashboards
- **Deployed on Render** as a production web service with a lightweight, serverless‑friendly architecture

---

## 🧠 Project Pipeline Architecture

┌──────────────────────────────┐
│ Raw Telco Dataset            │
└──────────────┬───────────────┘
               │
        [Data Validation]
               │
┌──────────────▼───────────────┐
│ Preprocessing Pipeline       │
│ • Missing value imputation   │
│ • One‑Hot Encoding           │
│ • Standard Scaling           │
│ • SMOTE (on train only)      │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│ Model Training               │
│ LogisticRegression (lbfgs)   │
│ RandomForestClassifier       │
│ GridSearchCV for tuning      │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│ Serialised Artifacts         │
│ model_lr.pkl, model_rf.pkl   │
│ scaler.pkl, encoder.pkl      │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│ Streamlit Application        │
│ (app.py)                     │
│ • Sidebar navigation         │
│ • Single‑/batch prediction   │
│ • Model performance dashboard│
│ • Feature importance charts  │
└──────────────┬───────────────┘
               │
         [Render Cloud]

### Data Flow in Prediction Mode

1. **User** uploads CSV or fills form → Streamlit sends data to `app.py`
2. App **loads** the trained pipeline (encoder + scaler + model) from disk
3. Preprocessing **transforms** raw input identically to training
4. Model **predicts** churn probability and class
5. Results **displayed** as table, probability histogram, and actionable insights

---

## 📂 Repository Structure

CUSTOMER_CHURN_PRED/
├── .github/                      # (inferred from ci-cd.yml & retrain.yml, though files appear at root)
├── .idea/                        # IDE folder
├── .pytest_cache/                # Root pytest cache
├── .streamlit/                   # Streamlit config directory
├── .dockerignore
├── .flake8
├── .gitignore
├── .python-version
├── CACHEDIR.TAG
├── Dockerfile
├── README.md
├── ci-cd.yml
├── config.toml
├── pytest.ini
├── render.yaml
├── requirements.txt
├── retrain.yml
├── start.sh
├── streamlit_start.sh
├── artifacts/
│   ├── model_comparison_report.txt
│   └── model_comparison_roc_curves.png
├── data/
│   ├── processed/
│   └── raw/
├── logs/
│   └── training.log
├── models/
│   └── production_model.pkl
├── src/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── api.py
│   ├── config.py
│   ├── data_pipeline.py
│   ├── hyperparameters.py
│   ├── model_comparison.py
│   ├── monitor.py
│   ├── pipeline.py
│   ├── retrain.py
│   ├── schemas.py
│   └── train.py
├── streamlit/
│   └── app.py
├── tests/
│   ├── __pycache__/
│   ├── .pytest_cache/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_data_pipeline.py
└── venv/                         # Virtual environment


---

## ⚙️ Tech Stack & Justification

 **Data processing** : Pandas, NumPy, Scikit‑learn | Battle‑tested, minimal dependencies 
 **Imbalance handling** : SMOTE (imbalanced‑learn) | Boosted recall by 5% with negligible overfitting risk on this dataset size 
 **Modelling** : Logistic Regression, Random Forest | Balance between interpretability and accuracy – see [model tradeoffs](docs/model_tradeoffs.md) 
 **UI** : Streamlit | Fastest path from notebook to interactive dashboard; no front‑end code 
 **Deployment** : Render | Free tier, auto‑deploy from GitHub, native Python support 

---

## 🏁 Quick Start (Local)

```bash
git clone https://github.com/Lokeshcoder1/customer_churn_pred.git
cd customer_churn_pred
pip install -r requirements.txt
python train_model.py   # optional – regenerates .pkl files
streamlit run app.py




