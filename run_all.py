"""Orchestrator: generate -> validate -> features -> models -> metrics.

Run from the project root:
    ~/workspace/quant-ipo-engine/.venv/bin/python run_all.py

Deterministic: seed 42 everywhere. Hold-outs are time-based (no
leakage) for Modules 1 and 2; Module 3 uses a stratified 80/20
random split with seed 42. Module 1 hold-out: Jul-Dec 2025
(branch x product x month). Module 2 hold-out: applications on or
after 2025-07-01. Every metric in the PDF and deck comes from
models/*.csv written here — nothing is hard-coded.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import charts as ch  # noqa: E402
import data_generation as dg  # noqa: E402
import data_pipeline as dp  # noqa: E402
import feature_engineering as fe  # noqa: E402
import models as md  # noqa: E402

MODELS = os.path.join(ROOT, "models")
CHARTS = os.path.join(ROOT, "reports", "charts")
PROC = os.path.join(ROOT, "data", "processed")
M1_HOLDOUT = pd.Timestamp("2025-07-01")
M2_HOLDOUT = pd.Timestamp("2025-07-01")

os.makedirs(MODELS, exist_ok=True)
os.makedirs(CHARTS, exist_ok=True)


def align(train_f, test_f):
    """Add zero columns so train and test share exactly the same
    feature set, then sort columns identically."""
    for col in train_f.columns:
        if col not in test_f.columns:
            test_f[col] = 0
    for col in test_f.columns:
        if col not in train_f.columns:
            train_f[col] = 0
    cols = sorted(train_f.columns)
    return train_f[cols], test_f[cols]


def main():
    print("== 1. generate synthetic data ==")
    dg.main()
    print("== 2. validate + clean ==")
    dp.main()

    pipe = pd.read_csv(os.path.join(PROC, "monthly_pipeline.csv"),
                       parse_dates=["month"])
    bor = pd.read_csv(os.path.join(PROC, "borrowers.csv"),
                      parse_dates=["application_date"])
    cust = pd.read_csv(os.path.join(PROC, "customers.csv"))

    # ---------------- Module 1: origination forecasting ------------
    print("== 3. module 1 — origination forecasting ==")
    X_all, y_all, meta = fe.build_pipeline_features(pipe)
    tr = meta["month"] < M1_HOLDOUT
    te = ~tr
    X_train, X_test = align(X_all[tr].copy(), X_all[te].copy())
    y_train, y_test = y_all[tr], y_all[te]
    m1, imp1, rf1 = md.origination_models(X_train, y_train,
                                          X_test, y_test)
    pd.DataFrame(m1).to_csv(os.path.join(MODELS, "origination_metrics.csv"),
                            index=False)
    imp1.rename_axis("feature").reset_index().rename(
        columns={0: "importance"}).to_csv(
        os.path.join(MODELS, "origination_importance.csv"), index=False)
    lr1 = LinearRegression().fit(X_train, y_train)
    fc = pd.DataFrame({
        "month": meta.loc[te, "month"],
        "branch": meta.loc[te, "branch"],
        "product": meta.loc[te, "product"],
        "y_true": y_test.values,
        "y_pred_rf": rf1.predict(X_test),
        "y_pred_lr": lr1.predict(X_test),
    })
    fc.to_csv(os.path.join(MODELS, "origination_holdout.csv"),
              index=False)
    print(f"  hold-out {len(y_test)} branch-product-months "
          f"(Jul-Dec 2025)")
    for r in m1:
        print(f"  {r['model']}: WAPE={r['wape']:.2%} MAPE={r['mape']:.2%} "
              f"MAE=${r['mae']:,.0f} R2={r['r2']:.3f}")

    # ---------------- Module 2: affordability + product fit --------
    print("== 4. module 2 — affordability & product fit ==")
    Xb_all = fe.build_borrower_features(bor)
    btr = bor["application_date"] < M2_HOLDOUT
    bte = ~btr
    Xb_train, Xb_test = align(Xb_all[btr].copy(), Xb_all[bte].copy())
    # affordability band
    yb_train, yb_test = bor.loc[btr, "affordability_band"], \
        bor.loc[bte, "affordability_band"]
    rule_band = fe.affordability_rule_predict(Xb_test)
    m2a, imp2a, rf2a = md.band_models(Xb_train, yb_train, Xb_test,
                                      yb_test, rule_band)
    pd.DataFrame(m2a).to_csv(
        os.path.join(MODELS, "affordability_metrics.csv"), index=False)
    imp2a.rename_axis("feature").reset_index().rename(
        columns={0: "importance"}).to_csv(
        os.path.join(MODELS, "affordability_importance.csv"),
        index=False)
    scored = Xb_test.copy()
    scored["y_true_band"] = yb_test.values
    scored["y_pred_band_rf"] = rf2a.predict(Xb_test)
    scored["y_true_fit"] = bor.loc[bte, "product_fit"].values
    # product fit
    yf_train, yf_test = bor.loc[btr, "product_fit"], \
        bor.loc[bte, "product_fit"]
    rule_fit = fe.product_fit_rule_predict(Xb_test)
    m2b, imp2b, rf2b = md.fit_models(Xb_train, yf_train, Xb_test,
                                     yf_test, rule_fit)
    pd.DataFrame(m2b).to_csv(
        os.path.join(MODELS, "product_fit_metrics.csv"), index=False)
    imp2b.rename_axis("feature").reset_index().rename(
        columns={0: "importance"}).to_csv(
        os.path.join(MODELS, "product_fit_importance.csv"), index=False)
    scored["y_pred_fit_rf"] = rf2b.predict(Xb_test)
    scored.to_csv(os.path.join(MODELS, "borrower_scored.csv"),
                  index=False)
    print(f"  hold-out {len(yb_test)} borrowers "
          f"({yb_test.value_counts().to_dict()})")
    for r in m2a:
        print(f"  band {r['model']}: macro-F1={r['macro_f1']:.3f} "
              f"acc={r['accuracy']:.3f}")
    for r in m2b:
        print(f"  fit  {r['model']}: macro-F1={r['macro_f1']:.3f} "
              f"acc={r['accuracy']:.3f}")

    # ---------------- Module 3: cross-sell need --------------------
    print("== 5. module 3 — cross-sell need ==")
    Xc_all = fe.build_customer_features(cust)
    yc_all = cust["mortgage_need"].astype(int)
    idx = np.arange(len(cust))
    idx_tr, idx_te = train_test_split(
        idx, test_size=0.20, random_state=42, stratify=yc_all)
    Xc_train, Xc_test = align(Xc_all.iloc[idx_tr].copy(),
                              Xc_all.iloc[idx_te].copy())
    yc_train, yc_test = yc_all.iloc[idx_tr], yc_all.iloc[idx_te]
    rule_c = fe.crosssell_rule_predict(Xc_test)
    m3, imp3, rf3, proba_rf = md.crosssell_models(
        Xc_train, yc_train, Xc_test, yc_test, rule_c)
    pd.DataFrame(m3).to_csv(
        os.path.join(MODELS, "crosssell_metrics.csv"), index=False)
    imp3.rename_axis("feature").reset_index().rename(
        columns={0: "importance"}).to_csv(
        os.path.join(MODELS, "crosssell_importance.csv"), index=False)
    q = md.review_queue_metrics(yc_test.values, proba_rf)
    pd.DataFrame([q]).to_csv(os.path.join(MODELS, "review_queue.csv"),
                             index=False)
    top_idx = np.argsort(-proba_rf)[:20]
    top = cust.iloc[idx_te].iloc[top_idx].copy()
    top["need_probability_model"] = np.round(proba_rf[top_idx], 3)
    top.to_csv(os.path.join(MODELS, "review_queue_top20.csv"),
               index=False)
    print(f"  hold-out {len(yc_test)} customers "
          f"({int(yc_test.sum())} with mortgage need)")
    for r in m3:
        print(f"  {r['model']}: F1={r['f1']:.3f} "
              f"ROC-AUC={r['roc_auc']:.3f}")
    print(f"  review queue top-20: {q['need_in_top']}/20 truly need "
          f"a mortgage; recall of need {q['recall_of_need']:.2%}")

    # ---------------- Benchmark table ------------------------------
    print("== 6. benchmark ==")
    rf_m1 = next(r for r in m1 if r["model"] == "Random Forest")
    snaive = next(r for r in m1
                  if r["model"].startswith("Seasonal-naive"))
    rf_m2a = next(r for r in m2a if r["model"] == "Random Forest")
    rule_m2a = next(r for r in m2a
                    if r["model"].startswith("DTI-only"))
    rf_m2b = next(r for r in m2b if r["model"] == "Random Forest")
    rule_m2b = next(r for r in m2b
                    if r["model"].startswith("Simplified"))
    rf_m3 = next(r for r in m3 if r["model"] == "Random Forest")
    lr_m3 = next(r for r in m3
                 if r["model"] == "Logistic Regression")
    rule_m3 = next(r for r in m3
                   if r["model"].startswith("Rule baseline"))
    # best learned model per module (Module 3's winner is Logistic
    # Regression, not Random Forest — the need probability was
    # generated logistically, so the linear model recovers it best)
    best_m1 = min((r for r in m1 if "baseline" not in r["model"]),
                  key=lambda r: r["wape"])
    best_m2a = max((r for r in m2a if "rule" not in r["model"]),
                   key=lambda r: r["macro_f1"])
    best_m2b = max((r for r in m2b if "rule" not in r["model"]),
                   key=lambda r: r["macro_f1"])
    best_m3 = max((r for r in m3 if "baseline" not in r["model"]),
                  key=lambda r: r["f1"])
    bench = pd.DataFrame([
        {"module": "Origination forecasting",
         "baseline": "Seasonal-naive",
         "baseline_score": snaive["wape"],
         "best_model": best_m1["model"], "best_score": best_m1["wape"],
         "metric": "WAPE (lower is better)"},
        {"module": "Affordability band",
         "baseline": "DTI-only rule",
         "baseline_score": rule_m2a["macro_f1"],
         "best_model": best_m2a["model"],
         "best_score": best_m2a["macro_f1"], "metric": "Macro-F1"},
        {"module": "Product fit",
         "baseline": "Simplified product rule",
         "baseline_score": rule_m2b["macro_f1"],
         "best_model": best_m2b["model"],
         "best_score": best_m2b["macro_f1"], "metric": "Macro-F1"},
        {"module": "Cross-sell need",
         "baseline": "Rule baseline",
         "baseline_score": rule_m3["f1"],
         "best_model": best_m3["model"], "best_score": best_m3["f1"],
         "metric": "F1"},
        {"module": "Outreach queue (top 20)",
         "baseline": "Random queue",
         "baseline_score": q["base_rate"],
         "best_model": "Risk-ranked queue",
         "best_score": q["precision_at_n"],
         "metric": "Precision@20"},
    ])
    bench.to_csv(os.path.join(MODELS, "benchmark.csv"), index=False)
    print(bench.to_string(index=False))

    # ---------------- Charts ---------------------------------------
    print("== 7. charts ==")
    ch.originations_by_product(
        os.path.join(PROC, "monthly_pipeline.csv"),
        os.path.join(CHARTS, "originations_by_product.png"))
    ch.origination_actual_vs_forecast(
        os.path.join(PROC, "monthly_pipeline.csv"),
        os.path.join(MODELS, "origination_holdout.csv"),
        os.path.join(CHARTS, "origination_actual_vs_forecast.png"))
    ch.feature_importance(
        os.path.join(MODELS, "origination_importance.csv"),
        os.path.join(CHARTS, "feature_importance.png"),
        "Top drivers of monthly funded volume (Random Forest)")
    ch.model_comparison(
        os.path.join(MODELS, "affordability_metrics.csv"), "macro_f1",
        os.path.join(CHARTS, "affordability_model_comparison.png"),
        "Affordability band: macro-F1 by model (hold-out)")
    ch.product_fit_by_credit(
        os.path.join(PROC, "borrowers.csv"),
        os.path.join(CHARTS, "product_fit_by_credit.png"))
    ch.crosssell_lift(
        os.path.join(PROC, "customers.csv"),
        os.path.join(CHARTS, "crosssell_lift.png"))
    ch.correlation_heatmap(
        os.path.join(MODELS, "borrower_scored.csv"),
        os.path.join(CHARTS, "correlation_heatmap.png"))

    # ---------------- Model report ---------------------------------
    print("== 8. model report ==")
    lr_m1 = next(r for r in m1 if r["model"] == "Linear Regression")
    nv_m1 = next(r for r in m1 if r["model"].startswith("Naive baseline"))
    lr_m2a = next(r for r in m2a
                  if r["model"] == "Logistic Regression")
    lr_m2b = next(r for r in m2b
                  if r["model"] == "Logistic Regression")
    lr_m3 = next(r for r in m3
                 if r["model"] == "Logistic Regression")

    def _f(x, pct=False):
        if isinstance(x, float) and np.isnan(x):
            return "—"
        return f"{x:.2%}" if pct else f"{x:.3f}"

    report = f"""# Home Lending Pipeline & Mortgage Advisory Analytics — Model Report

**Role:** Sr. Home Lending Advisor, JPMorganChase, Larchmont NY —
onsite, full-time; apply by Dec 19, 2026. Base pay listed at
$15–17/hr; total compensation is commission-dependent and unstated.
**Author:** Bhupesh Chandra Dimri — **All data is synthetic. No real
borrower data, no real PII, no real JPMorgan Chase data.**
**Hold-outs:** Module 1 — Jul-Dec 2025 (time-based, {len(y_test)}
branch-product-months). Module 2 — applications on/after 2025-07-01
(time-based, {len(yb_test)} borrowers). Module 3 — stratified 80/20
split, seed 42 ({len(yc_test)} customers).
**Tests:** see `tests/run_tests.py` (run after this script).

## Module 1 — Origination forecasting (hold-out: {len(y_test)} rows)
Monthly funded volume per branch x product, from pipeline stages,
the 30-year rate, and seasonality. Features: lags 1/3/6/12, rolling
3-month mean, month/quarter seasonality, pipeline-stage counts,
rate, branch, product. WAPE is the primary metric: funded volume is
lumpy (spring peaks, rate moves), so MAPE explodes on
small-denominator branch-months — and MAPE here is computed over
non-zero actuals only, since some small branch-product-months fund
$0.
| Model | WAPE | MAPE | MAE | R² |
|---|---|---|---|---|
"""
    for r in m1:
        report += (f"| {r['model']} | {_f(r['wape'], True)} | "
                   f"{_f(r['mape'], True)} | "
                   f"${r['mae']:,.0f} | {_f(r['r2'])} |\n")
    report += f"""
The Random Forest ({_f(rf_m1['wape'], True)} WAPE) beats the
seasonal-naive baseline ({_f(snaive['wape'], True)}) by
{(snaive['wape'] - rf_m1['wape']) * 100:.1f} points and Linear
Regression ({_f(lr_m1['wape'], True)}). The seasonal-naive baseline
is weak by design: the rate path and the pipeline mix move year over
year, so last year's month does not predict this year's. Top
drivers: {', '.join(imp1.head(5).index.tolist())}.

## Module 2a — Affordability band (hold-out: {len(yb_test)} borrowers)
Three bands — Affordable / Stretch / Unaffordable — from a
transparent rule (back-DTI thresholds, credit band, down payment,
reserves) with 4% label noise. The DTI-only rule is the advisor's
first check; the full label rule also uses credit, down payment,
and reserves.
| Model | Macro-F1 | Accuracy | F1 Affordable | F1 Stretch | F1 Unaffordable |
|---|---|---|---|---|---|
"""
    for r in m2a:
        report += (f"| {r['model']} | {_f(r['macro_f1'])} | "
                   f"{_f(r['accuracy'])} | {_f(r['f1_Affordable'])} | "
                   f"{_f(r['f1_Stretch'])} | "
                   f"{_f(r['f1_Unaffordable'])} |\n")
    report += f"""
The Random Forest ({_f(rf_m2a['macro_f1'])} macro-F1) beats the
DTI-only rule ({_f(rule_m2a['macro_f1'])}) because affordability is
not a single ratio: a Fair-credit borrower with thin reserves is a
Stretch at a DTI the rule calls Affordable, and a Good-credit
borrower with 20% down and deep reserves is Affordable at a DTI
the rule calls Stretch. Top drivers:
{', '.join(imp2a.head(5).index.tolist())}.

## Module 2b — Product fit (hold-out: {len(yb_test)} borrowers)
Four-way fit — Conventional / FHA / VA / Jumbo — from a transparent
rule (veteran status, down payment, loan vs conforming/FHA limits,
credit band) with 4% label noise.
| Model | Macro-F1 | Accuracy | F1 Conventional | F1 FHA | F1 VA | F1 Jumbo |
|---|---|---|---|---|---|
"""
    for r in m2b:
        report += (f"| {r['model']} | {_f(r['macro_f1'])} | "
                   f"{_f(r['accuracy'])} | {_f(r['f1_Conventional'])} | "
                   f"{_f(r['f1_FHA'])} | {_f(r['f1_VA'])} | "
                   f"{_f(r['f1_Jumbo'])} |\n")
    report += f"""
The Random Forest ({_f(rf_m2b['macro_f1'])} macro-F1) beats the
simplified rule ({_f(rule_m2b['macro_f1'])}), which misses the
credit-band and FHA-ceiling interactions — for example, a Fair-credit
borrower with 5% down fits FHA, while the same borrower at 12% down
with Good credit fits Conventional. Top drivers:
{', '.join(imp2b.head(5).index.tolist())}.

**This is an advisory/education prototype on synthetic data, not a
lending decision tool.** The scorer explains its reasoning (DTI,
credit band, down payment, loan limits) so an advisor can walk a
customer through it. It makes no credit decision and uses no real
borrower data.

## Module 3 — Cross-sell need (hold-out: {len(yc_test)} customers)
Binary need: does this existing customer plausibly need a mortgage
product? The rule baseline is the advisor's first check — no
mortgage + tenure over 2 years + a triggering life event
(relocation, marriage, new child).
| Model | Precision | Recall | F1 | Accuracy | ROC-AUC |
|---|---|---|---|---|---|
"""
    for r in m3:
        report += (f"| {r['model']} | {_f(r['precision'])} | "
                   f"{_f(r['recall'])} | {_f(r['f1'])} | "
                   f"{_f(r['accuracy'])} | {_f(r['roc_auc'])} |\n")
    report += f"""
The {best_m3['model']} (F1 {_f(best_m3['f1'])}, ROC-AUC
{_f(best_m3['roc_auc'])}) beats the rule (F1 {_f(rule_m3['f1'])})
because need is a combination, not a checklist: balance, products
held, and income band move the probability even when the three rule
conditions are all met — or all absent. Notably, Logistic Regression
(F1 {_f(lr_m3['f1'])}, ROC-AUC {_f(lr_m3['roc_auc'])}) beats the
Random Forest (F1 {_f(rf_m3['f1'])}, ROC-AUC
{_f(rf_m3['roc_auc'])}): the need probability was generated
logistically from the observable features, so the linear model
recovers the structure and the forest mostly fits noise. That is
the honest result — the simpler model wins this module, and the
report says so. Top drivers:
{', '.join(imp3.head(5).index.tolist())}.

The outreach queue ranks hold-out customers by the model's need
probability: top 20 are {q['need_in_top']}/20 truly mortgage-need,
against a base rate of {q['base_rate']:.1%} — a random queue would
catch about {q['base_rate'] * 20:.1f}. Queue recall of all need
customers: {q['recall_of_need']:.1%}.

## Benchmark (best model vs interpretable baseline)
| Module | Baseline | Best model | Metric |
|---|---|---|---|
"""
    for _, b in bench.iterrows():
        report += (f"| {b['module']} | {b['baseline']} "
                   f"{b['baseline_score']:.3f} | {b['best_model']} "
                   f"{b['best_score']:.3f} | {b['metric']} |\n")
    report += """
## Limits (read before citing any number)
- All data is synthetic: branches, borrowers, customers, rates,
  prices, and outcomes are invented. Rates, lifts, and class shares
  are design choices, not JPMorgan Chase facts. A model that scores
  well here proves the *pipeline design*, not performance on a real
  mortgage book.
- Affordability and product-fit labels are generated from the same
  transparent rules the models learn, with 4% noise — the
  classifier's edge over the simple rule comes from the
  interactions the simple rule ignores, not from hidden signal.
  The scorer is an advisory prototype: it explains its reasoning
  and makes no credit decision.
- The cross-sell need probability is generated from observable
  features; on real customers, need is driven by life events and
  circumstances no model can see. The queue is a triage list, not
  a prospect list — the advisor still calls the customer.
- The candidate does not claim mortgage lending experience,
  retail banking sales experience, or FHA/VA/FNMA/FHLMC knowledge
  — all are [CONFIRM WITH CANDIDATE] items, not facts in this
  project. The base pay listed for this role is $15–17/hr with
  commission-dependent total compensation (unstated); it sits far
  below the candidate's $140–190K band.
"""
    with open(os.path.join(ROOT, "reports", "model_report.md"),
              "w") as fh:
        fh.write(report)
    print("  reports/model_report.md written")
    print("DONE. Next: python tests/run_tests.py")


if __name__ == "__main__":
    main()
