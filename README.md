# Large-Scale Credit Default Prediction Pipeline

## 📌 Executive Summary
This project implements a robust machine learning pipeline designed to predict the probability of future credit card payment defaults based on time-series behavioral and financial data. 

In a commercial portfolio setting, early identification of high-risk accounts is critical for proactive risk mitigation. By leveraging advanced gradient boosting techniques (XGBoost), this model identifies subtle shifts in spending patterns, delinquency rates, and payment histories to forecast default events before they occur.

## 🏢 Business Impact & ROI
*   **Risk Mitigation:** Enables early intervention strategies (e.g., credit limit adjustments, targeted engagement) to minimize financial exposure.
*   **Automated Insights:** Synthesizes raw transactional data into actionable business intelligence, highlighting the leading indicators of financial stress.
*   **Portfolio Analytics:** Provides a scalable framework for evaluating the overall health of a commercial credit portfolio.

## ⚙️ Technical Architecture
1.  **Data Generation & Wrangling:** Processes complex, multi-statement customer profiles (simulated).
2.  **Feature Engineering:** Extracts temporal trends (mean, max, min, std) across Delinquency, Spend, Payment, and Balance metrics.
3.  **Predictive Modeling:** Utilizes an optimized XGBoost Classifier, evaluated via ROC-AUC to handle severe class imbalance.
4.  **Reporting Engine:** Automatically generates feature importance visualizations and strategic executive summaries.

## 🚀 How to Run the Project

This repository includes a synthetic data generator so you can instantly test the pipeline without needing to download massive external datasets.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the Dataset
Creates a realistic, time-series financial dataset.
```bash
python generate_synthetic_data.py
```

### 3. Train the Model
Engineers features, trains the XGBoost model, and evaluates performance.
```bash
python train_xgboost_model.py
```

### 4. Generate Business Insights
Produces an automated executive summary and visualizes the leading indicators of default.
```bash
python business_insights_report.py
```

## 🧠 Note on Real-World Application
This code is architected to scale. The feature engineering pipeline and model hyperparameters are structurally identical to those used on massive-scale (50GB+) financial datasets, such as those featured in premier quantitative risk competitions.
