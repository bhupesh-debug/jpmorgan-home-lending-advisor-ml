"""Feature engineering for the three modules.

Module 1 — origination forecasting (branch x product x month):
    lags 1/3/6/12 of funded_volume, rolling 3-month mean, month
    seasonality, pipeline-stage counts (applications, pre-approvals,
    in-underwriting, approved), current 30-yr rate, branch, product.
    Every feature is knowable before the month closes — the model
    never sees the month's funded volume.
Module 2 — affordability band + product fit (per borrower):
    annual_income, monthly_debts, credit_score, down_payment_pct,
    home_price, loan_amount, front/back DTI, reserves_months, veteran,
    first_time_buyer, loan-to-limit ratios. IDs and dates excluded.
Module 3 — cross-sell need (per customer):
    checking_tenure_years, avg_balance (log), has_mortgage,
    life_event one-hots, products_held, age/income band one-hots.
    Customer IDs excluded.
"""

import numpy as np
import pandas as pd


def build_pipeline_features(pipe):
    """One row per branch x product x month, features + target."""
    pipe = pipe.sort_values(["branch", "product", "month"]).copy()
    g = pipe.groupby(["branch", "product"])["funded_volume"]
    pipe["lag1"] = g.shift(1)
    pipe["lag3"] = g.shift(3)
    pipe["lag6"] = g.shift(6)
    pipe["lag12"] = g.shift(12)
    pipe["roll3_mean"] = g.transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    pipe["month_num"] = pipe["month"].dt.month
    pipe["quarter"] = pipe["month"].dt.quarter
    pipe = pipe.dropna(subset=["lag1", "lag3", "lag6", "lag12"])
    num = ["lag1", "lag3", "lag6", "lag12", "roll3_mean", "month_num",
           "quarter", "rate_30y", "applications", "pre_approvals",
           "in_underwriting", "approved", "territory_home_price"]
    X = pipe[num].copy()
    X["branch"] = pipe["branch"].values
    X["product"] = pipe["product"].values
    X = pd.get_dummies(X, columns=["branch", "product"], dtype=float)
    return X, pipe["funded_volume"], pipe[["month", "branch", "product"]]


def build_borrower_features(bor):
    """Features for affordability band and product fit. The rule that
    generated the labels uses DTI, credit, down payment, and limits —
    the classifier must recover it from these same fields."""
    f = pd.DataFrame(index=bor.index)
    f["annual_income"] = bor["annual_income"]
    f["log_income"] = np.log(bor["annual_income"])
    f["monthly_debts"] = bor["monthly_debts"]
    f["credit_score"] = bor["credit_score"]
    f["down_payment_pct"] = bor["down_payment_pct"]
    f["home_price"] = bor["home_price"]
    f["loan_amount"] = bor["loan_amount"]
    f["front_dti"] = bor["front_dti"]
    f["back_dti"] = bor["back_dti"]
    f["reserves_months"] = bor["reserves_months"]
    f["veteran"] = bor["veteran"].astype(int)
    f["first_time_buyer"] = bor["first_time_buyer"].astype(int)
    f["loan_to_conforming"] = bor["loan_amount"] / 806_500
    f["loan_to_fha_limit"] = bor["loan_amount"] / 1_100_000
    f["ltv"] = 1 - bor["down_payment_pct"]
    f["debt_to_income"] = bor["monthly_debts"] / (bor["annual_income"] / 12)
    f["credit_band"] = bor["credit_band"].values
    f = pd.get_dummies(f, columns=["credit_band"], dtype=float)
    return f


def affordability_rule_predict(feats):
    """The advisor's first check: back-DTI thresholds only. The full
    label rule also uses credit band, down payment, and reserves, so
    this simple rule is the floor a model must beat."""
    pred = []
    for _, r in feats.iterrows():
        if r["back_dti"] > 0.50:
            pred.append("Unaffordable")
        elif r["back_dti"] > 0.43:
            pred.append("Stretch")
        else:
            pred.append("Affordable")
    return np.array(pred)


def product_fit_rule_predict(feats):
    """Simplified product rule: VA if veteran, Jumbo if over the
    conforming limit, FHA if down payment under 10%, else
    Conventional. The full label rule also uses credit bands and the
    FHA ceiling."""
    pred = []
    for _, r in feats.iterrows():
        if r["veteran"] == 1 and r["down_payment_pct"] < 0.10:
            pred.append("VA")
        elif r["loan_to_conforming"] > 1.0:
            pred.append("Jumbo")
        elif r["down_payment_pct"] < 0.10:
            pred.append("FHA")
        else:
            pred.append("Conventional")
    return np.array(pred)


def build_customer_features(cust):
    """Features for the cross-sell need classifier. IDs excluded."""
    f = pd.DataFrame(index=cust.index)
    f["checking_tenure_years"] = cust["checking_tenure_years"]
    f["log_balance"] = np.log(cust["avg_balance"])
    f["has_mortgage"] = cust["has_mortgage"].astype(int)
    f["no_mortgage"] = (~cust["has_mortgage"]).astype(int)
    f["products_held"] = cust["products_held"]
    f["tenure_over_2"] = (cust["checking_tenure_years"] > 2).astype(int)
    f["life_event"] = cust["life_event"].values
    f["age_band"] = cust["age_band"].values
    f["income_band"] = cust["income_band"].values
    f = pd.get_dummies(f, columns=["life_event", "age_band",
                                   "income_band"], dtype=float)
    return f


def crosssell_rule_predict(feats):
    """The advisor's first check: no mortgage + tenure over 2 years +
    a mortgage-triggering life event (relocation, marriage, new
    child)."""
    life = (feats.get("life_event_Relocation", 0)
            + feats.get("life_event_Marriage", 0)
            + feats.get("life_event_New child", 0))
    return ((feats["no_mortgage"] == 1)
            & (feats["tenure_over_2"] == 1)
            & (life >= 1)).astype(int).values
