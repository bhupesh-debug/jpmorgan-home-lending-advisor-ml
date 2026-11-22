"""Validation gates + cleaning for the synthetic home-lending data.

Gates fail the run on any violation: missing values, negative or
impossible money amounts, rates outside [5%, 10%], funnel counts
that increase downstream, DTI outside (0, 1.2], credit scores
outside [300, 850], loan amounts that exceed home price, or label
shares outside the design bounds. Cleaned copies land in
data/processed/ (identical to synthetic here — the gates are the
point, not a transformation).
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYN = os.path.join(ROOT, "data", "synthetic")
PROC = os.path.join(ROOT, "data", "processed")


def _check(cond, msg):
    if not cond:
        raise ValueError(f"VALIDATION GATE FAILED: {msg}")


def validate_pipeline(df):
    _check(df.isna().sum().sum() == 0, "pipeline has missing values")
    _check((df["rate_30y"] >= 5).all() and (df["rate_30y"] <= 10).all(),
           "rate outside [5, 10]")
    _check((df["applications"] >= 0).all(), "negative applications")
    _check((df["pre_approvals"] <= df["applications"]).all(),
           "pre-approvals exceed applications")
    _check((df["in_underwriting"] <= df["pre_approvals"]).all(),
           "in-underwriting exceeds pre-approvals")
    _check((df["approved"] <= df["in_underwriting"]).all(),
           "approved exceeds in-underwriting")
    _check((df["funded_count"] <= df["approved"]).all(),
           "funded exceeds approved")
    _check((df["funded_volume"] >= 0).all(), "negative funded volume")
    _check(df["month"].min() >= pd.Timestamp("2023-01-01")
           and df["month"].max() <= pd.Timestamp("2025-12-01"),
           "pipeline month out of range")
    _check(len(df) == 8 * 4 * 36, "pipeline row count != 1152")


def validate_borrowers(df):
    _check(df.isna().sum().sum() == 0, "borrowers have missing values")
    _check((df["annual_income"] > 0).all(), "non-positive income")
    _check((df["credit_score"] >= 300).all()
           and (df["credit_score"] <= 850).all(),
           "credit score outside [300, 850]")
    _check((df["down_payment_pct"] >= 0.03).all()
           and (df["down_payment_pct"] <= 0.25).all(),
           "down payment pct outside [0.03, 0.25]")
    _check((df["loan_amount"] <= df["home_price"]).all(),
           "loan amount exceeds home price")
    _check((df["back_dti"] > 0).all() and (df["back_dti"] <= 1.2).all(),
           "back DTI outside (0, 1.2]")
    _check((df["front_dti"] <= df["back_dti"]).all(),
           "front DTI exceeds back DTI")
    for col in ("affordability_band", "product_fit"):
        _check(df[col].isin(
            ["Affordable", "Stretch", "Unaffordable"] if col ==
            "affordability_band" else
            ["Conventional", "FHA", "VA", "Jumbo"]).all(),
            f"unexpected label in {col}")
    share = df["affordability_band"].value_counts(normalize=True)
    _check(0.30 < share.get("Affordable", 0) < 0.75,
           "Affordable share outside (0.30, 0.75)")
    _check(0.02 < share.get("Unaffordable", 0) < 0.30,
           "Unaffordable share outside (0.02, 0.30)")


def validate_customers(df):
    _check(df.isna().sum().sum() == 0, "customers have missing values")
    _check((df["checking_tenure_years"] >= 0).all(),
           "negative tenure")
    _check((df["avg_balance"] > 0).all(), "non-positive balance")
    _check((df["products_held"] >= 1).all(), "products_held < 1")
    _check(df["mortgage_need"].isin([True, False]).all(),
           "mortgage_need not boolean")
    rate = df["mortgage_need"].mean()
    _check(0.10 < rate < 0.60, "mortgage-need rate outside (0.10, 0.60)")


def main():
    os.makedirs(PROC, exist_ok=True)
    pipe = pd.read_csv(os.path.join(SYN, "monthly_pipeline.csv"),
                       parse_dates=["month"])
    bor = pd.read_csv(os.path.join(SYN, "borrowers.csv"),
                      parse_dates=["application_date"])
    cust = pd.read_csv(os.path.join(SYN, "customers.csv"))
    validate_pipeline(pipe)
    validate_borrowers(bor)
    validate_customers(cust)
    pipe.to_csv(os.path.join(PROC, "monthly_pipeline.csv"), index=False)
    bor.to_csv(os.path.join(PROC, "borrowers.csv"), index=False)
    cust.to_csv(os.path.join(PROC, "customers.csv"), index=False)
    print("  gates passed: pipeline, borrowers, customers")


if __name__ == "__main__":
    main()
