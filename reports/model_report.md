# Home Lending Pipeline & Mortgage Advisory Analytics — Model Report

**Role:** Sr. Home Lending Advisor, JPMorganChase, Larchmont NY —
onsite, full-time; apply by Dec 19, 2026. Base pay listed at
$15–17/hr; total compensation is commission-dependent and unstated.
**Author:** Bhupesh Chandra Dimri — **All data is synthetic. No real
borrower data, no real PII, no real JPMorgan Chase data.**
**Hold-outs:** Module 1 — Jul-Dec 2025 (time-based, 192
branch-product-months). Module 2 — applications on/after 2025-07-01
(time-based, 747 borrowers). Module 3 — stratified 80/20
split, seed 42 (1200 customers).
**Tests:** see `tests/run_tests.py` (run after this script).

## Module 1 — Origination forecasting (hold-out: 192 rows)
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
| Seasonal-naive baseline (same month last year) | 40.00% | 45.27% | $672,567 | 0.404 |
| Naive baseline (last month) | 38.96% | 44.64% | $655,013 | 0.419 |
| Linear Regression | 27.03% | 31.42% | $454,433 | 0.744 |
| Random Forest | 9.79% | 11.29% | $164,601 | 0.950 |

The Random Forest (9.79% WAPE) beats the
seasonal-naive baseline (40.00%) by
30.2 points and Linear
Regression (27.03%). The seasonal-naive baseline
is weak by design: the rate path and the pipeline mix move year over
year, so last year's month does not predict this year's. Top
drivers: in_underwriting, approved, roll3_mean, product_Jumbo, product_FHA.

## Module 2a — Affordability band (hold-out: 747 borrowers)
Three bands — Affordable / Stretch / Unaffordable — from a
transparent rule (back-DTI thresholds, credit band, down payment,
reserves) with 4% label noise. The DTI-only rule is the advisor's
first check; the full label rule also uses credit, down payment,
and reserves.
| Model | Macro-F1 | Accuracy | F1 Affordable | F1 Stretch | F1 Unaffordable |
|---|---|---|---|---|---|
| DTI-only rule baseline | 0.899 | 0.928 | 0.952 | 0.894 | 0.851 |
| Logistic Regression | 0.738 | 0.780 | 0.859 | 0.668 | 0.686 |
| Random Forest | 0.951 | 0.963 | 0.974 | 0.947 | 0.931 |

The Random Forest (0.951 macro-F1) beats the
DTI-only rule (0.899) because affordability is
not a single ratio: a Fair-credit borrower with thin reserves is a
Stretch at a DTI the rule calls Affordable, and a Good-credit
borrower with 20% down and deep reserves is Affordable at a DTI
the rule calls Stretch. Top drivers:
back_dti, front_dti, debt_to_income, credit_score, down_payment_pct.

## Module 2b — Product fit (hold-out: 747 borrowers)
Four-way fit — Conventional / FHA / VA / Jumbo — from a transparent
rule (veteran status, down payment, loan vs conforming/FHA limits,
credit band) with 4% label noise.
| Model | Macro-F1 | Accuracy | F1 Conventional | F1 FHA | F1 VA | F1 Jumbo |
|---|---|---|---|---|---|
| Simplified product rule baseline | 0.862 | 0.885 | 0.911 | 0.730 | 0.842 | 0.967 |
| Logistic Regression | 0.716 | 0.817 | 0.861 | 0.763 | 0.385 | 0.856 |
| Random Forest | 0.930 | 0.967 | 0.981 | 0.930 | 0.842 | 0.967 |

The Random Forest (0.930 macro-F1) beats the
simplified rule (0.862), which misses the
credit-band and FHA-ceiling interactions — for example, a Fair-credit
borrower with 5% down fits FHA, while the same borrower at 12% down
with Good credit fits Conventional. Top drivers:
veteran, down_payment_pct, ltv, loan_to_conforming, loan_to_fha_limit.

**This is an advisory/education prototype on synthetic data, not a
lending decision tool.** The scorer explains its reasoning (DTI,
credit band, down payment, loan limits) so an advisor can walk a
customer through it. It makes no credit decision and uses no real
borrower data.

## Module 3 — Cross-sell need (hold-out: 1200 customers)
Binary need: does this existing customer plausibly need a mortgage
product? The rule baseline is the advisor's first check — no
mortgage + tenure over 2 years + a triggering life event
(relocation, marriage, new child).
| Model | Precision | Recall | F1 | Accuracy | ROC-AUC |
|---|---|---|---|---|---|
| Rule baseline (no mortgage + tenure + event) | 0.644 | 0.295 | 0.405 | 0.677 | 0.599 |
| Logistic Regression | 0.540 | 0.767 | 0.634 | 0.670 | 0.744 |
| Random Forest | 0.544 | 0.456 | 0.496 | 0.655 | 0.690 |

The Logistic Regression (F1 0.634, ROC-AUC
0.744) beats the rule (F1 0.405)
because need is a combination, not a checklist: balance, products
held, and income band move the probability even when the three rule
conditions are all met — or all absent. Notably, Logistic Regression
(F1 0.634, ROC-AUC 0.744) beats the
Random Forest (F1 0.496, ROC-AUC
0.690): the need probability was generated
logistically from the observable features, so the linear model
recovers the structure and the forest mostly fits noise. That is
the honest result — the simpler model wins this module, and the
report says so. Top drivers:
log_balance, checking_tenure_years, products_held, has_mortgage, no_mortgage.

The outreach queue ranks hold-out customers by the model's need
probability: top 20 are 14/20 truly mortgage-need,
against a base rate of 37.2% — a random queue would
catch about 7.5. Queue recall of all need
customers: 3.1%.

## Benchmark (best model vs interpretable baseline)
| Module | Baseline | Best model | Metric |
|---|---|---|---|
| Origination forecasting | Seasonal-naive 0.400 | Random Forest 0.098 | WAPE (lower is better) |
| Affordability band | DTI-only rule 0.899 | Random Forest 0.951 | Macro-F1 |
| Product fit | Simplified product rule 0.862 | Random Forest 0.930 | Macro-F1 |
| Cross-sell need | Rule baseline 0.405 | Logistic Regression 0.634 | F1 |
| Outreach queue (top 20) | Random queue 0.372 | Risk-ranked queue 0.700 | Precision@20 |

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
