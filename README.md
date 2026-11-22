# Home Lending Pipeline & Mortgage Advisory Analytics

**ML portfolio project for the Sr. Home Lending Advisor role at JPMorganChase, Larchmont NY**

Author: Bhupesh Chandra Dimri · November 2026

**GitHub repo (to be created by the user):**
https://github.com/bhupesh-debug/jpmorgan-home-lending-advisor-ml

> **ALL DATA IS SYNTHETIC.** No real borrower data, no real PII, no real JPMorgan Chase data. Branches, borrowers, customers, rates, prices, and outcomes are invented with a fixed seed (42).

## What this project does

Three modules, all on synthetic data, all with time-based (or stratified) hold-outs and interpretable baselines:

1. **Origination pipeline forecasting** — monthly funded mortgage volume per branch × product, from pipeline stages, the 30-year rate, and seasonality. Random Forest vs seasonal-naive and naive baselines (WAPE is the primary metric).
2. **Borrower affordability & product-fit scorer** — transparent-rule affordability bands (Affordable / Stretch / Unaffordable) and product fit (Conventional / FHA / VA / Jumbo) from synthetic borrower profiles. **This is an advisory/education prototype, not a lending decision tool**: it explains its reasoning (DTI, credit band, down payment, loan limits) and makes no credit decision.
3. **Branch cross-sell analyzer** — which existing customers plausibly need a mortgage product, with a ranked top-20 outreach list and review queue.

## The honest gap

The candidate has 9+ years of FP&A, forecasting, and variance analysis at JPMorgan Chase, but does **not** claim mortgage lending experience, retail banking sales experience, or FHA/VA/FNMA/FHLMC knowledge — all are [CONFIRM WITH CANDIDATE] items. The base pay listed for this role is $15–17/hr with commission-dependent total compensation (unstated) — far below the candidate's $140–190K band; this is a backup-only role. This project proves pipeline design, affordability reasoning, and outreach analytics. It does not prove production mortgage experience.

No existing repo under github.com/bhupesh-debug maps to mortgage or home lending — confirmed by search Nov 22, 2026. Nothing is cited rather than forcing a fit.

## How to run

```bash
# from the project root, with the quant venv (or any Python 3.12+ env
# with the packages in requirements.txt):
python run_all.py            # generate -> validate -> model -> charts -> report
python tests/run_tests.py    # test suite (all checks must pass)
python build_pdf.py          # 10-page report -> ~/workspace/your_files/
python build_deck.py         # 10-slide deck -> ~/workspace/your_files/
```

## Results

See `reports/model_report.md` for the full numbers (all from `models/*.csv`, nothing hard-coded). Headline results (hold-outs):

- Module 1 — Random Forest origination forecast vs seasonal-naive baseline (WAPE).
- Module 2a — Random Forest affordability band vs DTI-only rule (macro-F1).
- Module 2b — Random Forest product fit vs simplified product rule (macro-F1).
- Module 3 — Random Forest cross-sell need vs rule baseline (F1, ROC-AUC); top-20 outreach queue precision vs base rate.

## Files

```
run_all.py  build_pdf.py  build_deck.py  make_notebooks.py
requirements.txt  README.md  .gitignore
src/            data_generation.py  data_pipeline.py
                feature_engineering.py  models.py  charts.py
tests/          run_tests.py
notebooks/      01_eda.ipynb  02_feature_engineering.ipynb
                03_modeling.ipynb  04_evaluation.ipynb
data/           synthetic/  processed/
models/         13 CSVs (metrics, hold-outs, queue, benchmark)
reports/        model_report.md  charts/ (7 PNGs)
```

## Limits (read before citing any number)

- All data is synthetic; rates, prices, class shares, and outcomes are design choices, not JPMorgan Chase facts.
- Affordability and product-fit labels come from the same transparent rules the models learn, with 4% noise — the classifier's edge over the simple rule comes from the interactions the simple rule ignores, not from hidden signal.
- The cross-sell need probability is generated from observable features; real need is driven by life events and circumstances no model can see.
- The scorer is an advisory prototype: it explains its reasoning and makes no credit decision. Do not use it to decide lending.
