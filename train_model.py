"""
train_model.py
--------------
Credit-card default prediction on the UCI "Default of Credit Card Clients" dataset
(30,000 real card holders, Taiwan, April-September 2005).

Target: does the customer default on next month's payment?

Pipeline:
  1. Load & clean          fix undocumented category codes, rename PAY_0 -> PAY_1
  2. Feature engineering   6-month behavioural history -> delinquency, utilisation,
                           repayment-ratio and trend features
  3. Models                Logistic Regression (scorecard-style baseline) vs XGBoost
  4. Evaluation            ROC-AUC, Gini, KS, PR-AUC, 5-fold CV, top-decile capture
  5. Decision threshold    chosen to minimise expected credit loss, not at 0.5
  6. Explainability        per-customer SHAP reason codes (XGBoost TreeSHAP)
  7. Artefacts             models/, reports/metrics.json, reports/scored_test_customers.csv

Run:
    python train_model.py
"""

import os, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

warnings.filterwarnings("ignore")

SEED = 42
DATA_PATH = os.path.join("data", "uci_credit_default.csv")
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Business assumptions for the threshold analysis (NT$, the dataset's currency).
# These are stated placeholders, not facts from the data.
# Both sides scale with the customer's outstanding balance (exposure).
LOSS_AVOIDED_SHARE = 0.30   # share of a defaulter's balance saved by early action
                            # (limit cut, collections call, restructuring)
MARGIN_LOST_SHARE  = 0.10   # share of a good customer's balance lost as a year of interest
                            # and fees when they are wrongly restricted and leave

MONTHS = range(1, 7)

# Plain-English reason codes, as a lender would show them to a credit officer
REASON_TEXT = {
    "delay_last": "Payment late last month", "delay_max": "Serious past delinquency",
    "delay_mean": "Frequently late", "months_delayed": "Many late months",
    "delay_trend": "Delinquency getting worse", "months_no_use": "Card rarely used",
    "util_last": "High utilisation now", "util_mean": "High average utilisation",
    "util_max": "Hit credit limit", "util_trend": "Utilisation rising",
    "over_limit_months": "Went over limit", "pay_ratio_last": "Paid little of last bill",
    "pay_ratio_mean": "Usually pays little of bill", "zero_pay_months": "Missed payments entirely",
    "pay_amt_mean": "Low repayment amounts", "bill_growth": "Balance growing fast",
    "limit_bal": "Low credit limit", "age": "Age", "education": "Education", "married": "Marital status",
    **{f"pay_status_{k}": f"Repayment status {k} month(s) ago" for k in range(1, 7)},
}        # 1 = September 2005 (most recent) ... 6 = April 2005


# ============================================================================
# 1. LOAD & CLEAN
# ============================================================================
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"PAY_0": "PAY_1", "default payment next month": "target"})
    # EDUCATION codes 0, 5, 6 and MARRIAGE code 0 are not in the data dictionary;
    # fold them into the documented "other" category.
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"]  = df["MARRIAGE"].replace({0: 3})
    assert df["ID"].is_unique and df.isnull().sum().sum() == 0
    return df


# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
def engineer_features(df):
    """Turn six months of statements into behavioural features.

    PAY_k     repayment status in month k: -2/-1/0 = paid or revolving on time,
              1..9 = payment delayed by that many months
    BILL_AMTk statement balance in month k
    PAY_AMTk  amount paid in month k (this pays off the bill of month k+1)
    """
    pay  = df[[f"PAY_{k}"     for k in MONTHS]].to_numpy()
    bill = df[[f"BILL_AMT{k}" for k in MONTHS]].to_numpy()
    paid = df[[f"PAY_AMT{k}"  for k in MONTHS]].to_numpy()
    limit = df["LIMIT_BAL"].to_numpy()[:, None]
    delay = np.clip(pay, 0, None)

    f = pd.DataFrame(index=df.index)
    # Delinquency
    f["delay_last"]        = delay[:, 0]
    f["delay_max"]         = delay.max(axis=1)
    f["delay_mean"]        = delay.mean(axis=1)
    f["months_delayed"]    = (delay > 0).sum(axis=1)
    f["delay_trend"]       = delay[:, :3].mean(axis=1) - delay[:, 3:].mean(axis=1)
    f["months_no_use"]     = (pay == -2).sum(axis=1)
    # Utilisation of the credit limit
    util = bill / limit
    f["util_last"]         = util[:, 0]
    f["util_mean"]         = util.mean(axis=1)
    f["util_max"]          = util.max(axis=1)
    f["util_trend"]        = util[:, :3].mean(axis=1) - util[:, 3:].mean(axis=1)
    f["over_limit_months"] = (util > 1).sum(axis=1)
    # Repayment behaviour: payment in month k against the bill it settles (month k+1)
    ratio = paid[:, :5] / np.where(bill[:, 1:] > 0, bill[:, 1:], np.nan)
    ratio = np.clip(ratio, 0, 2)
    f["pay_ratio_last"]    = np.nan_to_num(ratio[:, 0], nan=1.0)
    f["pay_ratio_mean"]    = np.nan_to_num(np.nanmean(np.where(np.isnan(ratio), np.nan, ratio), axis=1), nan=1.0)
    f["zero_pay_months"]   = ((paid == 0) & (np.hstack([bill[:, 1:], bill[:, -1:]]) > 0)).sum(axis=1)
    f["pay_amt_mean"]      = paid.mean(axis=1)
    f["bill_growth"]       = (bill[:, 0] - bill[:, 5]) / limit[:, 0]
    # Static profile
    f["limit_bal"]         = df["LIMIT_BAL"]
    f["age"]               = df["AGE"]
    f["education"]         = df["EDUCATION"]
    f["married"]           = (df["MARRIAGE"] == 1).astype(int)
    # Raw monthly statuses are kept: the most recent ones are strong signals on their own
    for k in MONTHS:
        f[f"pay_status_{k}"] = df[f"PAY_{k}"]
    return f


def ks_statistic(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def main():
    print("\n" + "=" * 60)
    print("  CREDIT CARD DEFAULT PREDICTION")
    print("=" * 60)
    df = load_data()
    X = engineer_features(df)
    y = df["target"]
    print(f"Loaded {len(df):,} card holders | default rate {y.mean():.1%} | {X.shape[1]} features")

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, df["ID"], test_size=0.2, random_state=SEED, stratify=y)
    exposure_test = df.loc[X_test.index, "BILL_AMT1"].clip(lower=0).to_numpy()

    # ------------------------------------------------------------------------
    # 3. MODELS
    # ------------------------------------------------------------------------
    models = {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")),
        ]),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.7, min_child_weight=5, reg_lambda=2.0,
            eval_metric="auc", random_state=SEED, n_jobs=-1,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}
    for name, model in models.items():
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, p)
        results[name] = {
            "model": model, "prob": p,
            "test_auc": auc, "gini": 2 * auc - 1, "ks": ks_statistic(y_test, p),
            "pr_auc": average_precision_score(y_test, p),
            "cv_auc_mean": cv_auc.mean(), "cv_auc_std": cv_auc.std(),
        }
        print(f"  {name:<20} AUC={auc:.4f}  Gini={2*auc-1:.3f}  KS={results[name]['ks']:.3f}  "
              f"CV AUC={cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")

    BEST, BASE = "XGBoost", "Logistic Regression"
    best = results[BEST]
    prob = best["prob"]
    yt = y_test.to_numpy()
    joblib.dump(models[BEST], "models/xgboost_default_model.pkl")
    joblib.dump(models[BASE], "models/logistic_baseline.pkl")

    # ------------------------------------------------------------------------
    # 4. RANKING QUALITY
    # ------------------------------------------------------------------------
    order = np.argsort(-prob)
    def capture(frac):
        k = int(round(len(prob) * frac))
        return yt[order[:k]].sum() / yt.sum()
    deciles = pd.DataFrame({"prob": prob, "y": yt})
    deciles["decile"] = pd.qcut(deciles["prob"].rank(method="first", ascending=False), 10, labels=range(1, 11))
    decile_table = (deciles.groupby("decile", observed=True)["y"].agg(["count", "mean", "sum"])
                    .rename(columns={"mean": "default_rate", "sum": "defaults"}))
    decile_table["cum_capture"] = decile_table["defaults"].cumsum() / yt.sum()

    # ------------------------------------------------------------------------
    # 5. LOSS-MINIMISING THRESHOLD
    # ------------------------------------------------------------------------
    def net_value(t):
        flag = prob >= t
        saved = (LOSS_AVOIDED_SHARE * exposure_test[flag & (yt == 1)]).sum()
        cost = (MARGIN_LOST_SHARE * exposure_test[flag & (yt == 0)]).sum()
        return saved - cost, int(flag.sum()), int((flag & (yt == 1)).sum())

    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    curve = [(t, *net_value(t)) for t in grid]
    t_opt, v_opt, n_opt, tp_opt = max(curve, key=lambda r: r[1])
    v_05, n_05, tp_05 = net_value(0.5)
    print(f"\n  Top 10% riskiest hold {capture(0.1):.1%} of defaults; top 20% hold {capture(0.2):.1%}")
    print(f"  Loss-minimising threshold {t_opt:.2f}: flag {n_opt} customers, catch {tp_opt} defaulters, "
          f"net NT${v_opt:,.0f} (vs NT${v_05:,.0f} at 0.50)")

    # ------------------------------------------------------------------------
    # 6. EXPLAINABILITY - SHAP reason codes
    # ------------------------------------------------------------------------
    booster = models[BEST].get_booster()
    contribs = booster.predict(xgb.DMatrix(X_test), pred_contribs=True)[:, :-1]   # drop bias column
    feat = np.array(X.columns)
    shap_importance = (pd.Series(np.abs(contribs).mean(axis=0), index=feat)
                       .sort_values(ascending=False))
    # Top 3 features pushing each customer's risk up, deduplicated by reason text
    reasons = []
    for row_c in contribs:
        texts = []
        for i in np.argsort(-row_c):
            if row_c[i] <= 0 or len(texts) == 3:
                break
            t = REASON_TEXT.get(feat[i], feat[i])
            if t not in texts:
                texts.append(t)
        reasons.append("; ".join(texts))

    # ------------------------------------------------------------------------
    # 7. ARTEFACTS
    # ------------------------------------------------------------------------
    scored = pd.DataFrame({
        "ID": id_test.to_numpy(), "default_prob": prob.round(4),
        "flagged": (prob >= t_opt).astype(int), "actual_default": yt,
        "exposure_ntd": exposure_test, "top_risk_drivers": reasons,
    }).sort_values("default_prob", ascending=False)
    scored.to_csv("reports/scored_test_customers.csv", index=False)
    shap_importance.round(5).rename("mean_abs_shap").to_csv("reports/shap_importance.csv")
    decile_table.round(4).to_csv("reports/decile_table.csv")
    pd.DataFrame(curve, columns=["threshold", "net_value_ntd", "flagged", "defaulters_caught"]) \
      .to_csv("reports/threshold_curve.csv", index=False)

    metrics = {
        "dataset": {"name": "UCI Default of Credit Card Clients", "rows": len(df),
                    "default_rate": round(float(y.mean()), 4), "features": X.shape[1],
                    "test_rows": len(y_test)},
        "models": {n: {k: round(float(r[k]), 4) for k in
                       ["test_auc", "gini", "ks", "pr_auc", "cv_auc_mean", "cv_auc_std"]}
                   for n, r in results.items()},
        "best_model": BEST, "baseline_model": BASE,
        "ranking": {"top10_capture": round(float(capture(0.1)), 4),
                    "top20_capture": round(float(capture(0.2)), 4),
                    "top_decile_default_rate": round(float(decile_table["default_rate"].iloc[0]), 4)},
        "threshold": {"assumptions": {"loss_avoided_share": LOSS_AVOIDED_SHARE,
                                      "margin_lost_share": MARGIN_LOST_SHARE},
                      "optimal": float(t_opt), "flagged": n_opt, "defaulters_caught": tp_opt,
                      "net_value_ntd": float(v_opt),
                      "at_0_5": {"flagged": n_05, "defaulters_caught": tp_05, "net_value_ntd": float(v_05)}},
        "top_features_shap": shap_importance.head(10).round(5).to_dict(),
    }
    with open("reports/metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("\n  Saved models/ and reports/ (metrics.json, scored_test_customers.csv, ...)")


if __name__ == "__main__":
    main()
