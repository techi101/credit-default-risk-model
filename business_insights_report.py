"""
business_insights_report.py
---------------------------
Builds charts and a credit-risk briefing from the outputs of train_model.py.
Every number is read from reports/; nothing is hard-coded.

Produces:
  reports/model_performance.png    ROC curves, decile default rates, threshold economics
  reports/shap_importance.png      top risk drivers (mean |SHAP|)
  reports/credit_risk_report.md    briefing for a credit-risk manager

Run:
    python business_insights_report.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = "reports"


def main():
    if not os.path.exists(f"{R}/metrics.json"):
        raise SystemExit("Run `python train_model.py` first.")
    with open(f"{R}/metrics.json") as f:
        m = json.load(f)
    deciles = pd.read_csv(f"{R}/decile_table.csv")
    curve = pd.read_csv(f"{R}/threshold_curve.csv")
    shap = pd.read_csv(f"{R}/shap_importance.csv", index_col=0)["mean_abs_shap"]
    scored = pd.read_csv(f"{R}/scored_test_customers.csv")
    best, base = m["best_model"], m["baseline_model"]
    bm, lm, th, rk = m["models"][best], m["models"][base], m["threshold"], m["ranking"]

    # ---------------------------------------------------------------- charts
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    avg = m["dataset"]["default_rate"]
    ax[0].bar(deciles["decile"], deciles["default_rate"] * 100,
              color=["#c0392b" if r > avg else "#2c7fb8" for r in deciles["default_rate"]])
    ax[0].axhline(avg * 100, color="grey", linestyle="--", label=f"portfolio average {avg:.0%}")
    ax[0].set(title="Actual default rate by risk decile (test set)",
              xlabel="Risk decile (1 = riskiest 10%)", ylabel="Default rate (%)")
    ax[0].set_xticks(deciles["decile"])
    ax[0].legend()

    ax[1].plot([0, *(deciles["decile"] * 10)], [0, *(deciles["cum_capture"] * 100)], marker="o", color="#2c7fb8",
               label=f"{best}")
    ax[1].plot([0, 100], [0, 100], "--", color="grey", label="random")
    ax[1].set(title="Share of defaulters caught vs share of customers reviewed",
              xlabel="% of customers reviewed (riskiest first)", ylabel="% of defaulters caught")
    ax[1].legend()

    ax[2].plot(curve["threshold"], curve["net_value_ntd"] / 1e6, color="#2c7fb8")
    ax[2].axvline(th["optimal"], color="#c0392b", linestyle="--", label=f"optimal {th['optimal']:.2f}")
    ax[2].axvline(0.5, color="grey", linestyle=":", label="default 0.50")
    ax[2].set(title="Net loss avoided vs decision threshold",
              xlabel="Default-probability threshold", ylabel="Net value (NT$ million)")
    ax[2].legend()
    for a in ax:
        a.grid(alpha=0.3)
        a.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f"{R}/model_performance.png", dpi=130)
    plt.close()

    top = shap.head(12).iloc[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(top.index, top.values, color="#2c7fb8")
    plt.title(f"Top risk drivers ({best}, mean |SHAP value|)")
    plt.xlabel("Average impact on predicted default (log-odds)")
    plt.grid(axis="x", alpha=0.3)
    plt.gca().set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f"{R}/shap_importance.png", dpi=130)
    plt.close()

    # ---------------------------------------------------------------- report
    flagged = scored[scored["flagged"] == 1]
    reason_counts = (flagged["top_risk_drivers"].str.split("; ").explode()
                     .value_counts().head(5))
    a = th["assumptions"]
    lines = [
        "# Credit Card Default Risk — Briefing",
        f"*{m['dataset']['rows']:,} real card holders (UCI Default of Credit Card Clients), "
        f"{m['dataset']['default_rate']:.1%} default rate. All figures are on a held-out test set "
        f"of {m['dataset']['test_rows']:,} customers.*",
        "",
        "## Model performance",
        "| Model | AUC | Gini | KS | PR-AUC | 5-fold CV AUC |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in m["models"].items():
        lines.append(f"| {name} | {r['test_auc']:.3f} | {r['gini']:.3f} | {r['ks']:.3f} | "
                     f"{r['pr_auc']:.3f} | {r['cv_auc_mean']:.3f} ± {r['cv_auc_std']:.3f} |")
    lines += [
        "",
        f"{best} improves Gini by **{bm['gini'] - lm['gini']:+.3f}** over the {base} baseline.",
        "",
        "## Ranking",
        f"- The riskiest 10% of customers default at **{rk['top_decile_default_rate']:.0%}**, "
        f"against {m['dataset']['default_rate']:.0%} overall.",
        f"- Reviewing the riskiest 10% catches **{rk['top10_capture']:.0%}** of all defaulters; "
        f"the riskiest 20% catches **{rk['top20_capture']:.0%}**.",
        "",
        "## Recommended action threshold",
        f"- Act on customers with predicted default probability ≥ **{th['optimal']:.2f}**: "
        f"{th['flagged']:,} customers flagged, {th['defaulters_caught']:,} actual defaulters caught.",
        f"- Net loss avoided: **NT${th['net_value_ntd']/1e6:.2f} million**, versus "
        f"NT${th['at_0_5']['net_value_ntd']/1e6:.2f} million with the default 0.50 cut-off "
        f"({th['at_0_5']['defaulters_caught']:,} defaulters caught).",
        f"- Assumptions: early action saves {a['loss_avoided_share']:.0%} of a defaulter's balance; "
        f"wrongly restricting a good customer loses {a['margin_lost_share']:.0%} of their balance in "
        f"margin. With these costs the break-even probability is "
        f"{a['margin_lost_share']/(a['loss_avoided_share']+a['margin_lost_share']):.2f}, which is "
        "close to the threshold the search found.",
        "",
        "## Why customers get flagged",
        "Most common reason codes among flagged customers (top 3 SHAP drivers per customer):",
        "",
        "| Reason | Flagged customers |",
        "|---|---|",
    ]
    lines += [f"| {r} | {c:,} |" for r, c in reason_counts.items()]
    lines += [
        "",
        "## Limits",
        "- Data is from one Taiwanese bank in 2005; a model for another market must be retrained.",
        "- Cost assumptions are placeholders; replace them with the lender's loss and margin figures.",
        "- Age, education and marital status are in the data. Many lending regulations restrict "
        "using such attributes, so a production model would need a fair-lending review.",
    ]
    with open(f"{R}/credit_risk_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n-> {R}/credit_risk_report.md, model_performance.png, shap_importance.png")


if __name__ == "__main__":
    main()
