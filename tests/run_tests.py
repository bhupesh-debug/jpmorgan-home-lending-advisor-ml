"""Test suite for Home Lending Pipeline & Mortgage Advisory Analytics.

Run from the project root AFTER `python run_all.py`:
    ~/workspace/quant-ipo-engine/.venv/bin/python tests/run_tests.py

Each `check(` call counts as one test at run time — several sit inside
loops and execute once per file, so the run total (68) exceeds the
call-site count (45). The PDF and deck report the run total. All
checks are deterministic; the data-generation determinism check
regenerates the synthetic files into two temp dirs and compares
hashes.
"""

import hashlib
import os
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import data_generation as dg  # noqa: E402
import data_pipeline as dp  # noqa: E402

SYN = os.path.join(ROOT, "data", "synthetic")
PROC = os.path.join(ROOT, "data", "processed")
MODELS = os.path.join(ROOT, "models")
CHARTS = os.path.join(ROOT, "reports", "charts")

PASSED = 0
FAILED = []


def check(cond, name):
    global PASSED
    if cond:
        PASSED += 1
    else:
        FAILED.append(name)
        print(f"  FAIL: {name}")


def _hash(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    print("== tests: files exist ==")
    for name in ("monthly_pipeline.csv", "borrowers.csv",
                 "customers.csv"):
        check(os.path.exists(os.path.join(SYN, name)),
              f"synthetic {name} exists")
        check(os.path.exists(os.path.join(PROC, name)),
              f"processed {name} exists")
    for name in ("origination_metrics.csv", "origination_importance.csv",
                 "origination_holdout.csv", "affordability_metrics.csv",
                 "affordability_importance.csv",
                 "product_fit_metrics.csv", "product_fit_importance.csv",
                 "borrower_scored.csv", "crosssell_metrics.csv",
                 "crosssell_importance.csv", "review_queue.csv",
                 "review_queue_top20.csv", "benchmark.csv"):
        check(os.path.exists(os.path.join(MODELS, name)),
              f"models/{name} exists")
    for name in ("originations_by_product.png",
                 "origination_actual_vs_forecast.png",
                 "feature_importance.png",
                 "affordability_model_comparison.png",
                 "product_fit_by_credit.png", "crosssell_lift.png",
                 "correlation_heatmap.png"):
        p = os.path.join(CHARTS, name)
        check(os.path.exists(p) and os.path.getsize(p) > 5000,
              f"chart {name} exists and non-trivial")
    check(os.path.exists(os.path.join(ROOT, "reports",
                                      "model_report.md")),
          "model report exists")
    with open(os.path.join(ROOT, "reports", "model_report.md")) as fh:
        rep = fh.read().lower().replace("\n", " ")
    check("synthetic" in rep,
          "model report states synthetic data")
    check("not a lending decision" in rep,
          "model report states scorer is not a lending decision tool")

    print("== tests: data shape + gates ==")
    pipe = pd.read_csv(os.path.join(PROC, "monthly_pipeline.csv"),
                       parse_dates=["month"])
    bor = pd.read_csv(os.path.join(PROC, "borrowers.csv"),
                      parse_dates=["application_date"])
    cust = pd.read_csv(os.path.join(PROC, "customers.csv"))
    check(len(pipe) == 1152, "pipeline has 1152 rows")
    check(len(bor) == 4500, "borrowers has 4500 rows")
    check(len(cust) == 6000, "customers has 6000 rows")
    check(pipe.isna().sum().sum() == 0, "pipeline has no nulls")
    check(bor.isna().sum().sum() == 0, "borrowers have no nulls")
    check(cust.isna().sum().sum() == 0, "customers have no nulls")
    check((pipe["pre_approvals"] <= pipe["applications"]).all(),
          "funnel: pre-approvals <= applications")
    check((pipe["approved"] <= pipe["in_underwriting"]).all(),
          "funnel: approved <= in-underwriting")
    check((pipe["funded_count"] <= pipe["approved"]).all(),
          "funnel: funded <= approved")
    check((pipe["rate_30y"] >= 5).all()
          and (pipe["rate_30y"] <= 10).all(),
          "rate within [5, 10]")
    check((bor["credit_score"] >= 300).all()
          and (bor["credit_score"] <= 850).all(),
          "credit scores within [300, 850]")
    check((bor["back_dti"] > 0).all()
          and (bor["back_dti"] <= 1.2).all(),
          "back DTI within (0, 1.2]")
    check((bor["loan_amount"] <= bor["home_price"]).all(),
          "loan amount <= home price")
    check(set(bor["affordability_band"].unique())
          <= {"Affordable", "Stretch", "Unaffordable"},
          "affordability labels are the three bands")
    check(set(bor["product_fit"].unique())
          <= {"Conventional", "FHA", "VA", "Jumbo"},
          "product-fit labels are the four products")
    check(0.10 < cust["mortgage_need"].mean() < 0.60,
          "mortgage-need rate within (0.10, 0.60)")
    # gates must pass on the processed files too
    try:
        dp.validate_pipeline(pipe)
        dp.validate_borrowers(bor)
        dp.validate_customers(cust)
        check(True, "validation gates pass on processed data")
    except ValueError as e:
        check(False, f"validation gates pass on processed data ({e})")

    print("== tests: hold-out integrity ==")
    m1 = pd.read_csv(os.path.join(MODELS, "origination_holdout.csv"),
                     parse_dates=["month"])
    check((m1["month"] >= pd.Timestamp("2025-07-01")).all(),
          "M1 hold-out starts 2025-07-01")
    check(len(m1) == 8 * 4 * 6, "M1 hold-out has 192 rows")
    check(m1["y_true"].min() >= 0, "M1 hold-out targets non-negative")
    scored = pd.read_csv(os.path.join(MODELS, "borrower_scored.csv"))
    check(len(scored) == int((bor["application_date"]
                              >= "2025-07-01").sum()),
          "M2 hold-out row count matches Jul-Dec 2025 applications")
    check(set(scored["y_true_band"].unique())
          <= {"Affordable", "Stretch", "Unaffordable"},
          "M2 scored bands are valid")
    top20 = pd.read_csv(os.path.join(MODELS, "review_queue_top20.csv"))
    check(len(top20) == 20, "review queue top-20 has 20 rows")
    check(top20["need_probability_model"].between(0, 1).all(),
          "queue probabilities within [0, 1]")

    print("== tests: metric sanity ==")
    m1m = pd.read_csv(os.path.join(MODELS, "origination_metrics.csv"))
    check((m1m["wape"] > 0).all() and (m1m["wape"] < 1).all(),
          "M1 WAPE within (0, 1)")
    check((m1m["r2"] > -1).all() and (m1m["r2"] <= 1).all(),
          "M1 R2 within (-1, 1]")
    rf_wape = float(m1m.loc[m1m["model"] == "Random Forest",
                           "wape"].iloc[0])
    sn_wape = float(m1m.loc[m1m["model"].str.startswith("Seasonal-naive"),
                           "wape"].iloc[0])
    check(rf_wape < sn_wape, "M1 RF beats seasonal-naive WAPE")
    m2a = pd.read_csv(os.path.join(MODELS, "affordability_metrics.csv"))
    check((m2a["macro_f1"] >= 0).all()
          and (m2a["macro_f1"] <= 1).all(),
          "M2a macro-F1 within [0, 1]")
    rf_f1 = float(m2a.loc[m2a["model"] == "Random Forest",
                         "macro_f1"].iloc[0])
    rule_f1 = float(m2a.loc[m2a["model"].str.startswith("DTI-only"),
                           "macro_f1"].iloc[0])
    check(rf_f1 >= rule_f1, "M2a RF beats DTI-only rule macro-F1")
    m2b = pd.read_csv(os.path.join(MODELS, "product_fit_metrics.csv"))
    rf_f1b = float(m2b.loc[m2b["model"] == "Random Forest",
                          "macro_f1"].iloc[0])
    rule_f1b = float(m2b.loc[m2b["model"].str.startswith("Simplified"),
                            "macro_f1"].iloc[0])
    check(rf_f1b >= rule_f1b, "M2b RF beats simplified rule macro-F1")
    m3 = pd.read_csv(os.path.join(MODELS, "crosssell_metrics.csv"))
    check((m3["roc_auc"] >= 0.5).all()
          and (m3["roc_auc"] <= 1).all(),
          "M3 ROC-AUC within [0.5, 1]")
    rf_f1c = float(m3.loc[m3["model"] == "Random Forest",
                         "f1"].iloc[0])
    lr_f1c = float(m3.loc[m3["model"] == "Logistic Regression",
                         "f1"].iloc[0])
    rule_f1c = float(m3.loc[m3["model"].str.startswith("Rule baseline"),
                          "f1"].iloc[0])
    check(max(rf_f1c, lr_f1c) >= rule_f1c,
          "M3 best learned model beats rule baseline F1")
    check(lr_f1c >= rf_f1c,
          "M3 Logistic Regression beats Random Forest (logistic data)")
    q = pd.read_csv(os.path.join(MODELS, "review_queue.csv")).iloc[0]
    check(q["precision_at_n"] >= q["base_rate"],
          "queue precision >= base rate")
    check(0 <= q["recall_of_need"] <= 1, "queue recall within [0, 1]")
    check(q["need_in_top"] >= 10, "queue top-20 catches >= 10 need")

    print("== tests: determinism ==")
    with tempfile.TemporaryDirectory() as d1, \
            tempfile.TemporaryDirectory() as d2:
        dg.main(out_dir=d1)
        dg.main(out_dir=d2)
        for name in ("monthly_pipeline.csv", "borrowers.csv",
                     "customers.csv"):
            check(_hash(os.path.join(d1, name))
                  == _hash(os.path.join(d2, name)),
                  f"deterministic generation: {name}")

    print(f"\n{PASSED} passed, {len(FAILED)} failed")
    if FAILED:
        print("FAILED:", FAILED)
        raise SystemExit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
