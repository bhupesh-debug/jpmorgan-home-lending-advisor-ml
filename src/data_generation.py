"""Synthetic data generation for Home Lending Pipeline & Mortgage
Advisory Analytics.

ALL DATA IS SYNTHETIC. No real borrower data, no real PII, no real
JPMorgan Chase data. Branches, borrowers, customers, rates, prices,
and outcomes are invented with a fixed seed (42). Nothing here is a
credit decision: Module 2 is an advisory/education prototype that
scores affordability and product fit from transparent rules on
synthetic profiles — it is NOT a lending decision tool.

Datasets (all in data/synthetic/):
- monthly_pipeline.csv — branch x product x month, Jan 2023-Dec 2025
  (8 branches x 4 products x 36 months = 1,152 rows)
- borrowers.csv — synthetic mortgage applicants with income, debts,
  credit band, down payment, home price, veteran / first-time flags
- customers.csv — synthetic existing bank customers for the
  cross-sell module (no mortgage held, or mortgage held elsewhere)

Design notes:
- The 30-year rate series is a smooth synthetic path (6.4%-7.6%)
  with seasonality; lower rates lift applications.
- Affordability and product-fit labels come from transparent rules
  (DTI thresholds, LTV/down-payment bands, credit bands, loan
  limits) with 4% label noise, so a classifier must recover the
  rule from the data rather than memorize it.
- Cross-sell need is a logistic-style probability of the observable
  features plus noise; the rule baseline uses only three of them.
"""

import os

import numpy as np
import pandas as pd

SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYN = os.path.join(ROOT, "data", "synthetic")

BRANCHES = {
    # branch: (base monthly applications, avg home price in territory)
    "Larchmont": (52, 1_150_000),
    "Scarsdale": (48, 1_250_000),
    "White Plains": (44, 780_000),
    "New Rochelle": (40, 720_000),
    "Bronxville": (34, 1_050_000),
    "Rye": (30, 1_300_000),
    "Mamaroneck": (28, 950_000),
    "Greenwich": (26, 1_600_000),
}

PRODUCTS = {
    # product: (share of applications, avg loan amount, pull-through)
    "Conventional": (0.46, 620_000, 0.80),
    "FHA": (0.24, 420_000, 0.74),
    "VA": (0.10, 480_000, 0.78),
    "Jumbo": (0.20, 1_150_000, 0.72),
}

CONFORMING_LIMIT = 806_500      # 2025 conforming loan limit (simplified)
FHA_LIMIT = 1_100_000           # high-cost-area FHA ceiling (simplified)

MONTHS = pd.period_range("2023-01", "2025-12", freq="M")


def _rate_series(rng):
    """Synthetic 30-year fixed rate path: slow drift + seasonal dip
    in spring, plus noise. Range roughly 6.4%-7.6%."""
    t = np.arange(len(MONTHS))
    base = 6.9 + 0.35 * np.sin(2 * np.pi * t / 12.0) \
        - 0.25 * np.sin(2 * np.pi * t / 36.0)
    noise = rng.normal(0, 0.12, len(MONTHS))
    return np.clip(base + noise, 6.2, 7.8)


def _seasonal_factor(month):
    """Spring buying season peaks; winter troughs."""
    return {1: 0.82, 2: 0.86, 3: 1.02, 4: 1.12, 5: 1.18, 6: 1.14,
            7: 1.05, 8: 1.00, 9: 0.96, 10: 0.94, 11: 0.88,
            12: 0.84}[month]


def generate_pipeline(rng):
    rates = _rate_series(rng)
    rows = []
    for mi, per in enumerate(MONTHS):
        rate = rates[mi]
        # lower rates lift applications; elasticity ~ -0.35 per point
        rate_mult = 1.0 - 0.35 * (rate - 6.9)
        season = _seasonal_factor(per.month)
        for branch, (base_apps, terr_price) in BRANCHES.items():
            for product, (share, avg_loan, pull) in PRODUCTS.items():
                lam = base_apps * share * rate_mult * season
                lam = max(lam, 0.5)
                apps = rng.poisson(lam)
                preapp = int(round(apps * rng.uniform(0.55, 0.70)))
                in_uw = int(round(preapp * rng.uniform(0.55, 0.75)))
                approved = int(round(in_uw * rng.uniform(0.80, 0.92)))
                funded = int(round(approved * pull
                                   * rng.uniform(0.92, 1.05)))
                funded = max(funded, 0)
                # average loan scales with territory price vs product
                loan_amt = avg_loan * (terr_price / 1_000_000) ** 0.35
                loan_amt *= rng.uniform(0.94, 1.06)
                funded_vol = funded * loan_amt
                rows.append({
                    "month": per.to_timestamp(),
                    "branch": branch,
                    "product": product,
                    "rate_30y": round(rate, 2),
                    "applications": apps,
                    "pre_approvals": preapp,
                    "in_underwriting": in_uw,
                    "approved": approved,
                    "funded_count": funded,
                    "avg_loan_amount": round(loan_amt, 0),
                    "funded_volume": round(funded_vol, 0),
                    "territory_home_price": terr_price,
                })
    return pd.DataFrame(rows)


def _pit_factor():
    """Rough monthly PITI per $ of LOAN amount (principal+interest+
    taxes+insurance, 30-yr, ~6.5-7%)."""
    return 0.0070


def generate_borrowers(rng, n=4500):
    app_months = rng.choice(MONTHS.to_timestamp(), size=n)
    income = np.round(rng.lognormal(np.log(148_000), 0.42, n), -3)
    income = np.clip(income, 55_000, 620_000)
    debts = np.round(income / 12 * rng.uniform(0.02, 0.16, n), 0)
    credit = rng.choice(["Excellent", "Good", "Fair", "Poor"], size=n,
                        p=[0.30, 0.42, 0.20, 0.08])
    credit_score = {
        "Excellent": rng.integers(740, 821, n),
        "Good": rng.integers(680, 740, n),
        "Fair": rng.integers(620, 680, n),
        "Poor": rng.integers(540, 620, n),
    }
    # vectorised per-row credit score via the band draw
    scores = np.array([int(credit_score[b][i])
                       for i, b in enumerate(credit)])
    down_pct = np.round(rng.uniform(0.03, 0.25, n), 3)
    # home price tied to income (2.8x-5.6x) so DTIs land in a
    # realistic band instead of exploding on cheap incomes
    multiple = rng.uniform(2.8, 5.6, n)
    home_price = np.round(income * multiple, -3)
    home_price = np.clip(home_price, 320_000, 2_800_000)
    veteran = rng.random(n) < 0.08
    first_time = rng.random(n) < 0.38
    reserves_months = rng.integers(1, 13, n)

    loan_amt = home_price * (1 - down_pct)
    piti = loan_amt * _pit_factor()
    front_dti = piti / (income / 12)
    back_dti = (piti + debts) / (income / 12)

    # ---- affordability band: transparent rule + 4% label noise ----
    band = []
    for i in range(n):
        if back_dti[i] > 0.50 or (credit[i] == "Poor"
                                  and down_pct[i] < 0.10):
            band.append("Unaffordable")
        elif back_dti[i] > 0.43 or (credit[i] == "Fair"
                                    and reserves_months[i] < 3):
            band.append("Stretch")
        else:
            band.append("Affordable")
    band = np.array(band)
    flip = rng.random(n) < 0.04
    choices = np.array(["Affordable", "Stretch", "Unaffordable"])
    band[flip] = rng.choice(choices, size=int(flip.sum()))

    # ---- product fit: transparent rule + 4% label noise ----
    fit = []
    for i in range(n):
        if veteran[i] and down_pct[i] < 0.10:
            fit.append("VA")
        elif loan_amt[i] > CONFORMING_LIMIT:
            fit.append("Jumbo")
        elif (down_pct[i] < 0.10 and credit[i] in ("Fair", "Good")
              and loan_amt[i] <= FHA_LIMIT):
            fit.append("FHA")
        else:
            fit.append("Conventional")
    fit = np.array(fit)
    flip = rng.random(n) < 0.04
    pchoices = np.array(["Conventional", "FHA", "VA", "Jumbo"])
    fit[flip] = rng.choice(pchoices, size=int(flip.sum()))

    return pd.DataFrame({
        "application_id": [f"APP-{100000 + i}" for i in range(n)],
        "application_date": app_months,
        "annual_income": income.astype(int),
        "monthly_debts": debts.astype(int),
        "credit_band": credit,
        "credit_score": scores,
        "down_payment_pct": down_pct,
        "home_price": home_price.astype(int),
        "loan_amount": np.round(loan_amt, 0).astype(int),
        "veteran": veteran,
        "first_time_buyer": first_time,
        "reserves_months": reserves_months,
        "front_dti": np.round(front_dti, 3),
        "back_dti": np.round(back_dti, 3),
        "affordability_band": band,
        "product_fit": fit,
    })


def generate_customers(rng, n=6000):
    tenure = np.round(rng.uniform(0.2, 18.0, n), 1)
    balance = np.round(rng.lognormal(np.log(9_500), 0.9, n), 0)
    balance = np.clip(balance, 250, 250_000)
    has_mortgage = rng.random(n) < 0.42
    life_event = rng.choice(
        ["No event", "Marriage", "New child", "Relocation", "Retirement",
         "Job change"], size=n,
        p=[0.52, 0.09, 0.10, 0.12, 0.06, 0.11])
    products_held = rng.integers(1, 7, n)
    age_band = rng.choice(["25-34", "35-44", "45-54", "55-64", "65+"],
                          size=n, p=[0.18, 0.30, 0.26, 0.16, 0.10])
    income_band = rng.choice(["<75k", "75-125k", "125-200k", "200k+"],
                             size=n, p=[0.22, 0.34, 0.30, 0.14])
    # logistic-style need probability from observable features
    logit = (-2.6
             + 1.5 * (~has_mortgage).astype(float)
             + 0.9 * np.isin(life_event,
                             ["Relocation", "Marriage", "New child"])
             + 0.5 * (tenure > 2).astype(float)
             + 0.4 * (products_held >= 3).astype(float)
             + 0.3 * np.isin(income_band, ["125-200k", "200k+"])
             - 0.6 * (age_band == "65+").astype(float))
    prob = 1 / (1 + np.exp(-logit))
    need = rng.random(n) < prob
    return pd.DataFrame({
        "customer_id": [f"CUST-{200000 + i}" for i in range(n)],
        "checking_tenure_years": tenure,
        "avg_balance": balance.astype(int),
        "has_mortgage": has_mortgage,
        "life_event": life_event,
        "products_held": products_held,
        "age_band": age_band,
        "income_band": income_band,
        "mortgage_need": need,
        "need_probability": np.round(prob, 3),
    })


def main(out_dir=None):
    out_dir = out_dir or SYN
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pipe = generate_pipeline(rng)
    bor = generate_borrowers(rng)
    cust = generate_customers(rng)
    pipe.to_csv(os.path.join(out_dir, "monthly_pipeline.csv"),
                index=False)
    bor.to_csv(os.path.join(out_dir, "borrowers.csv"), index=False)
    cust.to_csv(os.path.join(out_dir, "customers.csv"), index=False)
    print(f"  monthly_pipeline.csv: {len(pipe)} rows "
          f"({pipe['funded_volume'].sum() / 1e6:,.0f}M funded)")
    print(f"  borrowers.csv: {len(bor)} rows "
          f"({bor['affordability_band'].value_counts().to_dict()})")
    print(f"  customers.csv: {len(cust)} rows "
          f"({int(cust['mortgage_need'].sum())} with mortgage need)")


if __name__ == "__main__":
    main()
