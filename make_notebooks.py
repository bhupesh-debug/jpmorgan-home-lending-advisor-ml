"""Generate the four notebooks (01_eda, 02_feature_engineering,
03_modeling, 04_evaluation) as .ipynb files with markdown + code
cells. Run from the project root AFTER `python run_all.py`:
    ~/workspace/quant-ipo-engine/.venv/bin/python make_notebooks.py
"""

import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(ROOT, "notebooks")
os.makedirs(NB, exist_ok=True)


def nb(cells):
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3",
                       "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4, "nbformat_minor": 5}


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [],
            "source": text.splitlines(True)}


NOTEBOOKS = {
    "01_eda.ipynb": [
        md("# 01 — Exploratory Data Analysis\n\n**All data is "
           "synthetic.** No real borrower data, no real PII, no real "
           "JPMorgan Chase data.\n\nThree datasets: monthly pipeline "
           "(branch x product x month), borrowers (synthetic "
           "applicants), customers (existing bank customers)."),
        code("import pandas as pd\n"
             "pipe = pd.read_csv('data/processed/monthly_pipeline.csv', "
             "parse_dates=['month'])\n"
             "bor = pd.read_csv('data/processed/borrowers.csv', "
             "parse_dates=['application_date'])\n"
             "cust = pd.read_csv('data/processed/customers.csv')\n"
             "print(pipe.shape, bor.shape, cust.shape)\n"
             "print(pipe.describe().T)\n"
             "print(bor['affordability_band'].value_counts())\n"
             "print(bor['product_fit'].value_counts())\n"
             "print(cust['mortgage_need'].mean())"),
        md("## Funnel check\n\nPre-approvals <= applications <= "
           "in-underwriting <= approved <= funded, every row. Rates "
           "stay in [5, 10]; DTI in (0, 1.2]; credit scores in "
           "[300, 850]."),
        code("assert (pipe['pre_approvals'] <= pipe['applications']).all()\n"
             "assert (pipe['funded_count'] <= pipe['approved']).all()\n"
             "assert bor['back_dti'].between(0, 1.2).all()\n"
             "print('funnel + bounds OK')"),
    ],
    "02_feature_engineering.ipynb": [
        md("# 02 — Feature Engineering\n\nModule 1: lags 1/3/6/12, "
           "rolling 3-month mean, month/quarter seasonality, "
           "pipeline-stage counts, rate, branch, product — all "
           "knowable before the month closes.\n\nModule 2: income, "
           "debts, credit score, down payment, home price, loan "
           "amount, front/back DTI, reserves, veteran, first-time, "
           "loan-to-limit ratios, LTV.\n\nModule 3: tenure, log "
           "balance, has/no mortgage, life event, products held, "
           "age/income bands."),
        code("import sys\nsys.path.insert(0, 'src')\n"
             "import pandas as pd\n"
             "import feature_engineering as fe\n"
             "pipe = pd.read_csv('data/processed/monthly_pipeline.csv', "
             "parse_dates=['month'])\n"
             "X, y, meta = fe.build_pipeline_features(pipe)\n"
             "print(X.shape, 'features; target:', y.name)\n"
             "print(list(X.columns)[:12])"),
    ],
    "03_modeling.ipynb": [
        md("# 03 — Modeling\n\nModule 1: seasonal-naive | naive | "
           "Linear Regression | Random Forest (WAPE primary).\n\n"
           "Module 2: DTI-only / simplified product rule | Logistic "
           "Regression | Random Forest (macro-F1).\n\nModule 3: rule "
           "baseline | Logistic Regression | Random Forest (F1, "
           "ROC-AUC) + top-20 review queue."),
        code("import pandas as pd\n"
             "print(pd.read_csv('models/origination_metrics.csv'))\n"
             "print(pd.read_csv('models/affordability_metrics.csv'))\n"
             "print(pd.read_csv('models/product_fit_metrics.csv'))\n"
             "print(pd.read_csv('models/crosssell_metrics.csv'))\n"
             "print(pd.read_csv('models/benchmark.csv'))"),
    ],
    "04_evaluation.ipynb": [
        md("# 04 — Evaluation\n\nHold-outs: Module 1 Jul-Dec 2025 "
           "(time-based); Module 2 applications on/after 2025-07-01 "
           "(time-based); Module 3 stratified 80/20, seed 42.\n\n"
           "The honest reading: labels are generated from transparent "
           "rules with 4% noise, so the classifier's edge over the "
           "simple rule comes from the interactions the simple rule "
           "ignores — not hidden signal. The Module 2 scorer is an "
           "advisory prototype, not a lending decision tool."),
        code("import pandas as pd\n"
             "print(pd.read_csv('models/review_queue.csv'))\n"
             "top = pd.read_csv('models/review_queue_top20.csv')\n"
             "print(top[['customer_id', 'life_event', "
             "'checking_tenure_years', 'need_probability_model']].head())\n"
             "print(open('reports/model_report.md').read()[:2000])"),
    ],
}

for name, cells in NOTEBOOKS.items():
    with open(os.path.join(NB, name), "w") as fh:
        json.dump(nb(cells), fh, indent=1)
    print("wrote", os.path.join(NB, name))
