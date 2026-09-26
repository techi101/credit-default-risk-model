# 💳 Credit Card Default Risk Model

> **Real data (30,000 card holders) · XGBoost vs logistic scorecard · AUC 0.780 · Gini 0.561 · KS 0.437 · SHAP reason codes · loss-minimising cut-off**

Predicts which credit-card customers will default on next month's payment, using six months of their statement history. It then turns the scores into a decision: who to act on, and why each customer was flagged.

Dataset: [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (Yeh & Lien, 2009). 30,000 real customers of a Taiwanese bank, April–September 2005, 22.1% default rate. CC BY 4.0.

---

## 🎯 Results (held-out test set, 6,000 customers)

| Model | AUC | Gini | KS | PR-AUC | 5-fold CV AUC |
|---|---|---|---|---|---|
| Logistic Regression (scorecard baseline) | 0.751 | 0.503 | 0.388 | 0.515 | 0.770 ± 0.004 |
| **XGBoost** | **0.780** | **0.561** | **0.437** | **0.567** | **0.788 ± 0.005** |

*Gini (= 2·AUC − 1) and KS are the standard ranking metrics in credit risk. For reference, published results on this dataset are typically around AUC 0.78.*

- **Ranking:** the riskiest 10% of customers default at **70%**, against 22% overall. Reviewing that 10% catches **32%** of all defaulters, and the riskiest 20% catches **51%**.
- **Decision threshold:** acting on customers with P(default) ≥ **0.26** flags 1,563 customers and catches **791** defaulters. At the usual 0.50 cut-off it catches only 492. Net loss avoided is **NT$8.77M vs NT$8.00M***.
- **Explainability:** each customer gets their top 3 risk drivers from SHAP values, in plain English (e.g. *"Payment late last month; Serious past delinquency; Frequently late"*). These work like the adverse-action reason codes lenders are required to give.

\*Assumes early action saves 30% of a defaulter's balance, and wrongly restricting a good customer loses 10% of their balance in margin. With those costs the break-even probability is 0.25, which matches the threshold the search found.

![Model performance](reports/model_performance.png)
![SHAP drivers](reports/shap_importance.png)

---

## ⚙️ How it works

1. **Cleaning.** `PAY_0` is renamed to `PAY_1` (a known naming error in the source). Undocumented `EDUCATION` codes (0/5/6) and `MARRIAGE` code 0 are folded into "other".
2. **Feature engineering.** The 6-month history becomes 26 behavioural features:
   | Group | Examples |
   |---|---|
   | Delinquency | last / max / mean months late, months delayed, delinquency trend |
   | Utilisation | balance ÷ limit (last, mean, max, trend), months over limit |
   | Repayment | payment ÷ the bill it settles, months with zero payment, average payment |
   | Profile | credit limit, age, education, marital status |
3. **Models.** A logistic regression (the traditional scorecard approach) vs XGBoost, with 5-fold stratified CV on the training set.
4. **Threshold.** The cut-off is chosen by sweeping thresholds and minimising expected loss, with costs that scale with each customer's balance.
5. **Explainability.** XGBoost's built-in TreeSHAP (`pred_contribs`) gives global feature importance and per-customer reason codes.

## 🗂️ Structure

```
├── train_model.py                 # cleaning → features → models → threshold → SHAP → reports/
├── business_insights_report.py    # charts + credit-risk briefing from reports/
├── data/uci_credit_default.csv    # source data
└── reports/
    ├── metrics.json               # every score and business figure
    ├── credit_risk_report.md      # briefing for a credit-risk manager
    ├── scored_test_customers.csv  # probability, flag and reason codes per test customer
    ├── decile_table.csv, threshold_curve.csv, shap_importance.csv
    └── model_performance.png, shap_importance.png
```

## 🚀 Run

```bash
pip install -r requirements.txt
python train_model.py
python business_insights_report.py
```
Runs on a laptop CPU in under a minute.

## ⚠️ Limitations

- One bank, one country, 2005. The feature pipeline transfers to other markets; the fitted model does not.
- Cost figures are placeholders; plug in a lender's real loss-given-default and margin numbers.
- Age, education and marital status are predictive but restricted in many lending regimes; a production model would need a fair-lending review, and possibly retraining without them.
