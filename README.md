# Vendor Invoice Intelligence System
**Freight Cost Prediction & Invoice Risk Flagging**

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Business Objectives](#business-objectives)
- [Data Sources](#data-sources)
- [Exploratory Data Analysis](#exploratory-data-analysis-eda)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [End-to-End Application](#end-to-end-application)
- [Project Structure](#project-structure)
- [How to Run This Project](#how-to-run-this-project)

---

## 🎯 Project Overview

This project implements an **end-to-end machine learning system** designed to support finance teams by:

1. **Predicting expected freight cost** for vendor invoices
2. **Flagging high-risk invoices** that require manual review

---

## 💼 Business Objectives

- ✅ Reduce financial risk through early detection of anomalies
- ✅ Improve operational efficiency by automating low-risk approvals
- ✅ Enable data-driven decision-making for vendor negotiations

---

## 🗄️ Data Sources

Data stored in SQLite database with tables:
- **vendor_invoice** – Invoice-level financial data
- **purchases** – Item-level purchase details
- **purchase_prices** – Reference prices
- **inventory snapshots** – begin/end inventory

---

## 📊 EDA

Business-driven questions:
- Do flagged invoices have higher financial exposure?
- Does freight scale linearly with quantity?
- Statistical tests confirm flagged vs normal invoice differences

---

## 🤖 Models Used

### Freight Prediction (Regression)
- Linear Regression (baseline)
- Decision Tree Regressor
- **Random Forest Regressor** (final)

### Invoice Flagging (Classification)
- Logistic Regression (baseline)
- Decision Tree Classifier
- **Random Forest Classifier with GridSearchCV** (final)

---

## 📈 Evaluation Metrics

**Freight:** MAE, RMSE, R² Score  
**Flagging:** Accuracy, Precision, Recall, F1-score (89% accuracy)

---

## 🎬 End-to-End Application

Streamlit dashboard with:
- Interactive model selection
- Real-time predictions
- Human-readable outputs

---

## 📁 Project Structure
inventory-invoice-analytics/
├── data/
│   └── inventory.db
├── freight_cost_prediction/
├── invoice_flagging/
├── inference/
│   ├── predict_freight.py
│   ├── predict_invoice_flag.py
│   └── app.py
├── README.md
└── requirements.txt

---

## 🚀 How to Run This Project

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/vendor-invoice-analytics.git
```

### 2. Train Models
```bash
python freight_cost_prediction/train_freight_model.py
python invoice_flagging/train.py
```

### 3. Test Models
```bash
python inference/predict_freight.py
python inference/predict_invoice_flag.py
```

### 4. Open Application
```bash
streamlit run inference/app.py
```

Visit `http://localhost:8501`

---

## 📝 Example Usage

```python
from inference.predict_freight import predict_freight_cost

input_data = {"Quantity": [100], "Dollars": [18500]}
result = predict_freight_cost(input_data)
print(result)
```

---


---

**Last Updated:** July 2026  
