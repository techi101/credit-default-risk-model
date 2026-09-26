# Credit Card Default Risk — Briefing
*30,000 real card holders (UCI Default of Credit Card Clients), 22.1% default rate. All figures are on a held-out test set of 6,000 customers.*

## Model performance
| Model | AUC | Gini | KS | PR-AUC | 5-fold CV AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.751 | 0.503 | 0.388 | 0.515 | 0.770 ± 0.004 |
| XGBoost | 0.780 | 0.561 | 0.437 | 0.567 | 0.788 ± 0.005 |

XGBoost improves Gini by **+0.058** over the Logistic Regression baseline.

## Ranking
- The riskiest 10% of customers default at **70%**, against 22% overall.
- Reviewing the riskiest 10% catches **32%** of all defaulters; the riskiest 20% catches **51%**.

## Recommended action threshold
- Act on customers with predicted default probability ≥ **0.29** (chosen on training data; results below are on the test set): 1,409 customers flagged, 750 actual defaulters caught.
- Net loss avoided: **NT$8.62 million**, versus NT$8.00 million with the default 0.50 cut-off (492 defaulters caught).
- Assumptions: early action saves 30% of a defaulter's balance; wrongly restricting a good customer loses 10% of their balance in margin. With these costs the break-even probability is 0.25, which is close to the threshold chosen on the training data.

## Why customers get flagged
Most common reason codes among flagged customers (top 3 SHAP drivers per customer):

| Reason | Flagged customers |
|---|---|
| Serious past delinquency | 1,200 |
| Frequently late | 677 |
| Payment late last month | 634 |
| Low repayment amounts | 324 |
| Repayment status 1 month(s) ago | 261 |

## Limits
- Data is from one Taiwanese bank in 2005; a model for another market must be retrained.
- Cost assumptions are placeholders; replace them with the lender's loss and margin figures.
- Age, education and marital status are in the data. Many lending regulations restrict using such attributes, so a production model would need a fair-lending review.