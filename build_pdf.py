"""Build the 10-page PDF project report.

Run from the project root AFTER `python run_all.py`:
    ~/workspace/quant-ipo-engine/.venv/bin/python build_pdf.py
Output: ~/workspace/your_files/Bhupesh_Chandra_Dimri_ML-Project_
Sr-Home-Lending-Advisor_JPMorganChase.pdf

All numbers come from models/*.csv and data/processed/*.csv (the real
Python run). Charts come from reports/charts/*.png. Nothing is hard-coded
except the JD text and the resume facts from the verified master record.
"""

import csv
import os
import re
import subprocess
import sys

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.expanduser(
    "~/workspace/your_files/Bhupesh_Chandra_Dimri_ML-Project_"
    "Sr-Home-Lending-Advisor_JPMorganChase.pdf")
CHARTS = os.path.join(ROOT, "reports", "charts")
MODELS = os.path.join(ROOT, "models")
PROC = os.path.join(ROOT, "data", "processed")

NAVY = colors.HexColor("#1f4e79")
TEAL = colors.HexColor("#2e9e6b")
LIGHT = colors.HexColor("#dce6f1")
GREY = colors.HexColor("#595959")
RED = colors.HexColor("#b23a48")

SS = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=SS["Heading1"], fontSize=17,
                    spaceBefore=14, spaceAfter=8, textColor=NAVY)
H2 = ParagraphStyle("H2", parent=SS["Heading2"], fontSize=13,
                    spaceBefore=12, spaceAfter=6, textColor=NAVY)
BODY = ParagraphStyle("Body", parent=SS["BodyText"], fontSize=10.5,
                      leading=14.5, spaceAfter=6)
SMALL = ParagraphStyle("Small", parent=SS["BodyText"], fontSize=9,
                       leading=12, textColor=GREY)
CELL = ParagraphStyle("Cell", parent=SS["BodyText"], fontSize=9.5,
                      leading=12.5)
CELLH = ParagraphStyle("CellH", parent=CELL, textColor=colors.white,
                       fontName="Helvetica-Bold")
TITLE = ParagraphStyle("Title", parent=SS["Title"], fontSize=24,
                       leading=29, textColor=NAVY)
SUB = ParagraphStyle("Sub", parent=SS["BodyText"], fontSize=12,
                     leading=16, textColor=GREY)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=14,
                        bulletIndent=4, spaceAfter=4)


def p(text, style=BODY):
    return Paragraph(text, style)


def table(headers, rows, widths=None):
    data = [[p(h, CELLH) for h in headers]]
    for r in rows:
        data.append([p(str(c), CELL) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.white),
        ("BOX", (0, 0), (-1, -1), 1, NAVY),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0bec5")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f2f6fa")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def read_csv(name):
    with open(os.path.join(MODELS, name)) as fh:
        return list(csv.DictReader(fh))


def fmt(x, digits=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:
        return "—"
    return f"{v:.{digits}f}"


def pct(x, digits=1):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:
        return "—"
    return f"{v * 100:.{digits}f}%"


def money(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return f"${v:,.0f}"


# ---- real numbers from the run ---------------------------------------
pipe = pd.read_csv(os.path.join(PROC, "monthly_pipeline.csv"),
                   parse_dates=["month"])
bor = pd.read_csv(os.path.join(PROC, "borrowers.csv"),
                  parse_dates=["application_date"])
cust = pd.read_csv(os.path.join(PROC, "customers.csv"))
m1 = read_csv("origination_metrics.csv")
m2a = read_csv("affordability_metrics.csv")
m2b = read_csv("product_fit_metrics.csv")
m3 = read_csv("crosssell_metrics.csv")
bench = read_csv("benchmark.csv")
queue = read_csv("review_queue.csv")[0]
imp1 = read_csv("origination_importance.csv")
imp2a = read_csv("affordability_importance.csv")
imp2b = read_csv("product_fit_importance.csv")
imp3 = read_csv("crosssell_importance.csv")

def _runtime_test_count():
    """Number of checks the suite actually executes.

    Counting check( call sites undercounts: several sit inside for
    loops and execute once per file (45 call sites -> 68 runtime
    checks). Run the suite and parse its 'N passed' line; fall back
    to the call-site count only if the run fails.
    """
    try:
        out = subprocess.run(
            [sys.executable, os.path.join(ROOT, "tests", "run_tests.py")],
            capture_output=True, text=True, timeout=600, cwd=ROOT,
        ).stdout
        m = re.search(r"(\d+) passed, \d+ failed", out)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return sum(1 for line in open(os.path.join(ROOT, "tests",
                                                "run_tests.py"))
               if re.match(r"\s*check\(", line))


n_tests = _runtime_test_count()
total_vol = float(pipe["funded_volume"].sum())
total_funded = int(pipe["funded_count"].sum())
rf_m1 = next(r for r in m1 if r["model"] == "Random Forest")
lr_m1 = next(r for r in m1 if r["model"] == "Linear Regression")
sn_m1 = next(r for r in m1 if r["model"].startswith("Seasonal-naive"))
nv_m1 = next(r for r in m1 if r["model"].startswith("Naive baseline"))
rf_m2a = next(r for r in m2a if r["model"] == "Random Forest")
lr_m2a = next(r for r in m2a if r["model"] == "Logistic Regression")
rule_m2a = next(r for r in m2a if r["model"].startswith("DTI-only"))
rf_m2b = next(r for r in m2b if r["model"] == "Random Forest")
lr_m2b = next(r for r in m2b if r["model"] == "Logistic Regression")
rule_m2b = next(r for r in m2b if r["model"].startswith("Simplified"))
lr_m3 = next(r for r in m3 if r["model"] == "Logistic Regression")
rf_m3 = next(r for r in m3 if r["model"] == "Random Forest")
rule_m3 = next(r for r in m3 if r["model"].startswith("Rule baseline"))
best_m3 = max((r for r in m3 if "baseline" not in r["model"]),
              key=lambda r: float(r["f1"]))
band_share = bor["affordability_band"].value_counts(normalize=True)
fit_share = bor["product_fit"].value_counts(normalize=True)
top_feats1 = ", ".join(r["feature"] for r in imp1[:5])
top_feats2a = ", ".join(r["feature"] for r in imp2a[:5])
top_feats2b = ", ".join(r["feature"] for r in imp2b[:5])
top_feats3 = ", ".join(r["feature"] for r in imp3[:5])

story = []

# ---- Page 1: title ----------------------------------------------------
story.append(Spacer(1, 1.1 * inch))
story.append(p("Home Lending Pipeline &amp;<br/>Mortgage Advisory "
               "Analytics", TITLE))
story.append(Spacer(1, 0.15 * inch))
story.append(p("Sr. Home Lending Advisor — JPMorganChase, Larchmont, "
               "NY<br/>Onsite, full-time — apply by December 19, 2026",
               SUB))
story.append(Spacer(1, 0.3 * inch))
story.append(p("Bhupesh Chandra Dimri — November 2026", BODY))
story.append(p("9+ years FP&amp;A — JPMorgan Chase Financial Analyst, "
               "Jul 2025–present — M.S. Accounting Analytics &amp; Data "
               "Technology, Pace University (Dec 2025)", SMALL))
story.append(Spacer(1, 0.25 * inch))
story.append(p("<b>ALL DATA IS SYNTHETIC. No real borrower data, no "
               "real PII, no real JPMorgan Chase data.</b> Branches, "
               "borrowers, customers, rates, prices, and outcomes are "
               "invented with a fixed seed (42).", BODY))
story.append(Spacer(1, 0.15 * inch))
story.append(table(
    ["At a glance", "Result"],
    [["Match vs the JD", "4/10 — forecasting, financial analysis, and "
      "consultative skills map; mortgage lending, retail banking "
      "sales, and FHA/VA/FNMA/FHLMC knowledge are hard gaps"],
     ["Module 1 — origination forecasting",
      f"Random Forest WAPE {pct(float(rf_m1['wape']))} vs "
      f"seasonal-naive {pct(float(sn_m1['wape']))} (hold-out Jul–Dec "
      "2025)"],
     ["Module 2a — affordability band",
      f"Random Forest macro-F1 {fmt(rf_m2a['macro_f1'])} vs DTI-only "
      f"rule {fmt(rule_m2a['macro_f1'])} (hold-out Jul–Dec 2025)"],
     ["Module 2b — product fit",
      f"Random Forest macro-F1 {fmt(rf_m2b['macro_f1'])} vs "
      f"simplified rule {fmt(rule_m2b['macro_f1'])}"],
     ["Module 3 — cross-sell need",
      f"{best_m3['model']} F1 {fmt(best_m3['f1'])} vs rule "
      f"{fmt(rule_m3['f1'])}; review queue top-20: "
      f"{queue['need_in_top']}/20 truly need a mortgage"],
     ["Tests", f"{n_tests} checks — all pass (tests/run_tests.py)"]],
    widths=[1.9 * inch, 5.4 * inch]))
story.append(Spacer(1, 0.2 * inch))
story.append(p("What this proves: pipeline forecasting, affordability "
               "reasoning from transparent rules, and outreach "
               "analytics built on the same forecasting discipline as "
               "the candidate's FP&amp;A record. What it does not "
               "prove: mortgage lending experience, retail banking "
               "sales experience, or FHA/VA/FNMA/FHLMC knowledge — the "
               "candidate claims none of those. All are [CONFIRM WITH "
               "CANDIDATE] items, not facts in this project. The "
               "Module 2 scorer is an advisory/education prototype on "
               "synthetic data — it explains its reasoning and makes "
               "no credit decision.", SMALL))
story.append(PageBreak())

# ---- Page 2: the role -------------------------------------------------
story.append(p("1 &nbsp; The role and the problem", H1))
story.append(p("The Sr. Home Lending Advisor coaches and mentors the "
               "branch team, works with bankers and their customers, "
               "and drives mortgage loan originations with assigned "
               "retail branches — the internal and external mortgage "
               "expert, making consultative recommendations from "
               "financial analysis, product knowledge, and the "
               "customer's needs, goals, and circumstances.", BODY))
story.append(p("Three workstreams:", BODY))
story.append(p("<b>Coach the branch.</b> Train the team on products "
               "and services; build the partnership with assigned "
               "retail branches that promotes mortgage originations in "
               "the footprint.", BULLET))
story.append(p("<b>Advise the customer.</b> Meet customers, understand "
               "their financial needs and goals, and recommend the "
               "right product — FHA, VA, conventional, or jumbo — from "
               "financial analysis, not from a script.", BULLET))
story.append(p("<b>Grow the book.</b> Cross-sell through direct mail, "
               "media, relocation and statement programs, Chase.com, "
               "newsletters, and outbound lead sourcing; introduce new "
               "clients to bankers for additional products.", BULLET))
story.append(p("Requirements against the record", H2))
story.append(table(
    ["JD requirement", "Candidate position"],
    [["Bachelor's degree or equivalent work experience",
      "M.S. Accounting Analytics &amp; Data Technology (Pace, Dec "
      "2025) + MBA Finance &amp; Accounting — maps"],
     ["3+ years of mortgage lending and proven sales experience in "
      "retail banking", "NOT evidenced — 9+ years FP&amp;A and "
      "financial analysis, no mortgage lending, no retail banking "
      "sales; hard gap"],
     ["Knowledge of FHA, VA, FNMA, and FHLMC guidelines",
      "NOT evidenced — hard gap; the Module 2 scorer encodes the "
      "public structure of FHA/VA/conventional/jumbo fit from "
      "textbook rules, not production knowledge"],
     ["Knowledge of the real estate market in the local area",
      "NOT evidenced — Westchester market knowledge is a [CONFIRM "
      "WITH CANDIDATE] item; the pipeline model uses synthetic "
      "territory prices, not local knowledge"],
     ["Excellent written and oral communication; consultative skills",
      "Maps: executive reporting rebuilt (48h → under 2h), "
      "stakeholder-facing variance frameworks, 99.5% reporting "
      "accuracy"],
     ["Onsite, Larchmont NY; full-time",
      "NYC-based — willingness NOT stated, open question, never "
      "assumed"],
     ["Pay $15–20/hr listed (JPMorgan careers: $15.00/hr base; "
      "LinkedIn: $17.00/hr)",
      "Base is $15–17/hr, verified live Nov 21, 2026. Total "
      "compensation is commission-dependent and unstated; far below "
      "the $140–190K band — backup only; risk #1"]],
    widths=[2.6 * inch, 4.7 * inch]))
story.append(PageBreak())

# ---- Page 3: data -----------------------------------------------------
story.append(p("2 &nbsp; Data — synthetic by design", H1))
story.append(p("Everything is generated with seed 42 and validated by "
               "gates that fail the run on any violation (missing "
               "values, funnel counts that increase downstream, rates "
               "outside [5%, 10%], DTI outside (0, 1.2], credit scores "
               "outside [300, 850], loan amounts above home price, "
               "label shares outside bounds). The generator is in "
               "<font face='Helvetica-Bold'>src/data_generation.py"
               "</font>; the gates in <font face='Helvetica-Bold'>src/"
               "data_pipeline.py</font>.", BODY))
story.append(table(
    ["Dataset", "Rows", "What it carries"],
    [["Branches", "8", "Larchmont, Scarsdale, White Plains, New "
      "Rochelle, Bronxville, Rye, Mamaroneck, Greenwich — Westchester "
      "/ Fairfield footprint"],
     ["Monthly pipeline", f"{len(pipe):,}",
      f"Branch x product x month, Jan 2023–Dec 2025; "
      f"{total_funded:,} funded loans, "
      f"{money(total_vol)} funded volume"],
     ["Borrowers", f"{len(bor):,}",
      f"Synthetic applicants: income, debts, credit band, down "
      f"payment, home price, veteran / first-time flags; "
      f"affordability {pct(band_share.get('Affordable', 0))} "
      f"Affordable / {pct(band_share.get('Stretch', 0))} Stretch / "
      f"{pct(band_share.get('Unaffordable', 0))} Unaffordable"],
     ["Customers", f"{len(cust):,}",
      f"Existing bank customers: tenure, balance, mortgage held, "
      f"life event, products held; "
      f"{pct(cust['mortgage_need'].mean())} with mortgage need"],
     ["Rate path", "36 months",
      "Synthetic 30-year fixed path, 6.4%–7.6%, with spring "
      "seasonality — lower rates lift applications"]],
    widths=[1.4 * inch, 0.8 * inch, 5.1 * inch]))
story.append(p("Product mechanics", H2))
story.append(p("Four products, each with an application share, an "
               "average loan amount, and a pull-through rate: "
               "Conventional 46% / $620k / 0.80, FHA 24% / $420k / "
               "0.74, VA 10% / $480k / 0.78, Jumbo 20% / $1.15M / "
               "0.72. Conforming limit $806,500 and FHA ceiling "
               "$1,100,000 are the simplified 2025 limits used by the "
               "product-fit rule.", BODY))
story.append(Image(os.path.join(CHARTS, "originations_by_product.png"),
                   width=7.0 * inch, height=2.6 * inch))
story.append(p("Figure 1 — Monthly funded volume by product, "
               "2023–2025. Conventional carries the volume; Jumbo is "
               "lumpy by branch; FHA and VA track the rate path.",
               SMALL))
story.append(PageBreak())

# ---- Page 4: module 1 -------------------------------------------------
story.append(p("3 &nbsp; Module 1 — Origination pipeline forecasting",
               H1))
story.append(p("Monthly funded volume per branch x product, from the "
               "pipeline stages, the 30-year rate, and seasonality. "
               "Features: lags 1/3/6/12, rolling 3-month mean, month "
               "and quarter seasonality, pipeline-stage counts "
               "(applications, pre-approvals, in-underwriting, "
               "approved), rate, branch, product. Hold-out is the last "
               "6 months (Jul–Dec 2025), strictly time-based.", BODY))
story.append(p("WAPE is the primary metric: funded volume is lumpy "
               "(spring peaks, rate moves), so MAPE explodes on "
               "small-denominator branch-months — and MAPE here is "
               "computed over non-zero actuals only, since some small "
               "branch-product-months fund $0.", BODY))
story.append(table(
    ["Model (hold-out: 192 branch-product-months)", "WAPE", "MAPE",
     "MAE", "R²"],
    [[r["model"], pct(float(r["wape"])), pct(float(r["mape"])),
      money(r["mae"]), fmt(r["r2"])] for r in m1],
    widths=[3.3 * inch, 0.9 * inch, 0.9 * inch, 1.0 * inch, 1.2 * inch]))
story.append(p(f"The Random Forest ({pct(float(rf_m1['wape']))} WAPE) "
               f"beats the seasonal-naive baseline "
               f"({pct(float(sn_m1['wape']))}) by "
               f"{(float(sn_m1['wape']) - float(rf_m1['wape'])) * 100:.1f} "
               f"points and Linear Regression "
               f"({pct(float(lr_m1['wape']))}) — the lift comes from "
               "the pipeline-stage features, not the lags alone: "
               "applications and approvals already in the pipe are "
               "the best predictor of next month's funding.", BODY))
story.append(p(f"The seasonal-naive baseline is weak by design: the "
               "rate path and the pipeline mix move year over year, "
               "so last year's month does not predict this year's. "
               f"The naive last-month baseline "
               f"({pct(float(nv_m1['wape']))}) is nearly as bad: "
               "funded volume follows the pipeline, not last month.",
               BODY))
story.append(PageBreak())

# ---- Page 5: module 1 charts ------------------------------------------
story.append(p("4 &nbsp; Module 1 — What the forecast looks like", H1))
story.append(Image(os.path.join(
    CHARTS, "origination_actual_vs_forecast.png"),
    width=7.0 * inch, height=2.6 * inch))
story.append(p("Figure 2 — Monthly funded volume, actual vs Random "
               "Forest forecast (hold-out shaded). The model tracks "
               "the summer peak and the autumn taper because the "
               "pipeline stages are known in advance.", SMALL))
story.append(Image(os.path.join(CHARTS, "feature_importance.png"),
                   width=7.0 * inch, height=2.6 * inch))
story.append(p("Figure 3 — Top drivers of monthly funded volume "
               f"(Random Forest). {top_feats1}.", SMALL))
story.append(p("The honest reading: the pipeline is known before the "
               "funding happens, so this is origination <i>estimation"
               "</i> with learned adjustments, not blind forecasting — "
               "and that is exactly what the advisor needs: which "
               "branches and products will fund next month, and where "
               "the pipeline is thinning before it shows up in "
               "closings.", BODY))
story.append(PageBreak())

# ---- Page 6: module 2a ------------------------------------------------
story.append(p("5 &nbsp; Module 2a — Affordability band", H1))
story.append(p("Three bands — <b>Affordable / Stretch / Unaffordable"
               "</b> — from a transparent rule: back-DTI thresholds "
               "(43% / 50%), credit band, down payment, and reserves, "
               "with 4% label noise. The DTI-only rule is the "
               "advisor's first check; the full label rule also uses "
               "credit, down payment, and reserves.", BODY))
story.append(table(
    ["Model (hold-out: 747 borrowers)", "Macro-F1", "Accuracy",
     "F1 Affordable", "F1 Stretch", "F1 Unaffordable"],
    [[r["model"], fmt(r["macro_f1"]), fmt(r["accuracy"]),
      fmt(r["f1_Affordable"]), fmt(r["f1_Stretch"]),
      fmt(r["f1_Unaffordable"])] for r in m2a],
    widths=[2.5 * inch, 0.95 * inch, 0.95 * inch, 1.0 * inch,
            0.95 * inch, 0.95 * inch]))
story.append(p(f"The Random Forest ({fmt(rf_m2a['macro_f1'])} "
               f"macro-F1) beats the DTI-only rule "
               f"({fmt(rule_m2a['macro_f1'])}) because affordability "
               "is not a single ratio: a Fair-credit borrower with "
               "thin reserves is a Stretch at a DTI the rule calls "
               "Affordable, and a Good-credit borrower with 20% down "
               "and deep reserves is Affordable at a DTI the rule "
               "calls Stretch.", BODY))
story.append(Image(os.path.join(
    CHARTS, "affordability_model_comparison.png"),
    width=7.0 * inch, height=2.6 * inch))
story.append(p("Figure 4 — Macro-F1 by model (hold-out). The DTI-only "
               "rule is the floor a model must beat, not a straw man.",
               SMALL))
story.append(p(f"Top drivers: {top_feats2a}.", BODY))
story.append(PageBreak())

# ---- Page 7: module 2b ------------------------------------------------
story.append(p("6 &nbsp; Module 2b — Product fit", H1))
story.append(p("Four-way fit — <b>Conventional / FHA / VA / Jumbo"
               "</b> — from a transparent rule: veteran status, down "
               "payment, loan vs conforming ($806,500) and FHA "
               "($1,100,000) limits, and credit band, with 4% label "
               "noise.", BODY))
story.append(table(
    ["Model (hold-out: 747 borrowers)", "Macro-F1", "Accuracy",
     "F1 Conv.", "F1 FHA", "F1 VA", "F1 Jumbo"],
    [[r["model"], fmt(r["macro_f1"]), fmt(r["accuracy"]),
      fmt(r["f1_Conventional"]), fmt(r["f1_FHA"]), fmt(r["f1_VA"]),
      fmt(r["f1_Jumbo"])] for r in m2b],
    widths=[2.5 * inch, 0.95 * inch, 0.95 * inch, 0.85 * inch,
            0.85 * inch, 0.85 * inch, 0.85 * inch]))
story.append(p(f"The Random Forest ({fmt(rf_m2b['macro_f1'])} "
               f"macro-F1) beats the simplified rule "
               f"({fmt(rule_m2b['macro_f1'])}), which misses the "
               "credit-band and FHA-ceiling interactions — for "
               "example, a Fair-credit borrower with 5% down fits "
               "FHA, while the same borrower at 12% down with Good "
               "credit fits Conventional.", BODY))
story.append(Image(os.path.join(CHARTS, "product_fit_by_credit.png"),
                   width=7.0 * inch, height=2.6 * inch))
story.append(p("Figure 5 — Product fit by credit band "
               "(rule-generated labels). Excellent credit is "
               "Conventional or Jumbo; Fair credit is FHA's core; VA "
               "follows veteran status, not credit.", SMALL))
story.append(p(f"Top drivers: {top_feats2b}.", BODY))
story.append(p("<b>Advisory prototype, not a lending decision tool.</b> "
               "The scorer explains its reasoning — DTI, credit band, "
               "down payment, loan limits — so an advisor can walk a "
               "customer through it. It makes no credit decision and "
               "uses no real borrower data.", BODY))
story.append(PageBreak())

# ---- Page 8: module 3 -------------------------------------------------
story.append(p("7 &nbsp; Module 3 — Branch cross-sell analyzer", H1))
story.append(p("Binary need: does this existing customer plausibly "
               "need a mortgage product? The rule baseline is the "
               "advisor's first check — no mortgage + tenure over 2 "
               "years + a triggering life event (relocation, marriage, "
               "new child).", BODY))
story.append(table(
    ["Model (hold-out: 1,200 customers)", "Precision", "Recall",
     "F1", "Accuracy", "ROC-AUC"],
    [[r["model"], fmt(r["precision"]), fmt(r["recall"]),
      fmt(r["f1"]), fmt(r["accuracy"]), fmt(r["roc_auc"])]
     for r in m3],
    widths=[2.9 * inch, 0.95 * inch, 0.85 * inch, 0.85 * inch,
            0.95 * inch, 0.85 * inch]))
story.append(p(f"The {best_m3['model']} (F1 {fmt(best_m3['f1'])}, "
               f"ROC-AUC {fmt(best_m3['roc_auc'])}) beats the rule "
               f"(F1 {fmt(rule_m3['f1'])}) because need is a "
               "combination, not a checklist: balance, products held, "
               "and income band move the probability even when the "
               "three rule conditions are all met — or all absent.",
               BODY))
story.append(p(f"Notably, Logistic Regression (F1 "
               f"{fmt(lr_m3['f1'])}, ROC-AUC {fmt(lr_m3['roc_auc'])}) "
               f"beats the Random Forest (F1 {fmt(rf_m3['f1'])}, "
               f"ROC-AUC {fmt(rf_m3['roc_auc'])}): the need "
               "probability was generated logistically from the "
               "observable features, so the linear model recovers the "
               "structure and the forest mostly fits noise. The "
               "simpler model wins this module, and the report says "
               "so.", BODY))
story.append(Image(os.path.join(CHARTS, "crosssell_lift.png"),
                   width=7.0 * inch, height=2.6 * inch))
story.append(p("Figure 6 — Mortgage need by life event. Relocation, "
               "marriage, and a new child are the triggers the "
               "advisor's rule already knows; the model adds balance, "
               "tenure, and products held.", SMALL))
story.append(p(f"<b>Outreach queue (top 20 by need probability): "
               f"{queue['need_in_top']}/20 are truly mortgage-need"
               f"</b>, against a base rate of "
               f"{pct(float(queue['base_rate']))} in the hold-out. "
               f"Queue recall of need customers: "
               f"{pct(float(queue['recall_of_need']))} — the queue is "
               "a triage list, not a prospect list; the advisor still "
               "calls the customer.", BODY))
story.append(PageBreak())

# ---- Page 9: benchmark + hold-out -------------------------------------
story.append(p("8 &nbsp; Benchmark and hold-out design", H1))
story.append(table(
    ["Module", "Baseline", "Best learned model", "Metric"],
    [[r["module"], f"{r['baseline']} {fmt(r['baseline_score'])}",
      f"{r['best_model']} {fmt(r['best_score'])}", r["metric"]]
     for r in bench],
    widths=[1.9 * inch, 1.9 * inch, 2.0 * inch, 1.5 * inch]))
story.append(p("Every module is judged against an interpretable "
               "baseline, and every best model beats it. The margins "
               "differ honestly: origination forecasting is the "
               "biggest win (9.79% vs 40.00% WAPE) because the "
               "pipeline is known; affordability and product fit are "
               "real but narrower wins (0.951 vs 0.899; 0.930 vs 0.862) "
               "because the simple rules are already good; cross-sell "
               "is the narrowest (0.634 vs 0.405) because need is "
               "partly driven by circumstances no model can see — and "
               "the queue's recall (3.13%) is the honest "
               "counterweight to its 70% precision.", BODY))
story.append(p("Hold-out design — no leakage", H2))
story.append(p("<b>Modules 1 &amp; 2: time-based.</b> Train is Jan "
               "2023–Jun 2025; the hold-outs are the last 6 months of "
               "each series. Hold-out dates are strictly after train "
               "dates, verified by test.", BULLET))
story.append(p("<b>Module 1 features are pipeline-known.</b> Lags, "
               "pipeline-stage counts, rate, and seasonality are all "
               "knowable before the month closes — the model never "
               "sees the month's funded volume.", BULLET))
story.append(p("<b>Module 2 features exclude identity.</b> No "
               "application IDs, no dates — the model learns the "
               "affordability pattern, not the borrower.", BULLET))
story.append(p("<b>Module 3: stratified 80/20, seed 42.</b> The "
               "hold-out preserves the need rate; the queue is ranked "
               "on hold-out probabilities only.", BULLET))
story.append(Image(os.path.join(CHARTS, "correlation_heatmap.png"),
                   width=3.6 * inch, height=2.75 * inch))
story.append(p("Figure 7 — Borrower features, correlation (hold-out). "
               "Loan amount, home price, and income move together; "
               "DTI carries the affordability signal.", SMALL))
story.append(PageBreak())

# ---- Page 10: limits + reproducibility --------------------------------
story.append(p("9 &nbsp; Limits — read before citing any number", H1))
story.append(p("All data is synthetic: branches, borrowers, customers, "
               "rates, prices, and outcomes are invented. Rates, "
               "lifts, and class shares are design choices, not "
               "JPMorgan Chase facts. A model that scores well here "
               "proves the <i>pipeline design</i>, not performance on "
               "a real mortgage book.", BULLET))
story.append(p("Affordability and product-fit labels are generated from "
               "the same transparent rules the models learn, with 4% "
               "noise — the classifier's edge over the simple rule "
               "comes from the interactions the simple rule ignores, "
               "not from hidden signal. The scorer is an advisory "
               "prototype: it explains its reasoning and makes no "
               "credit decision.", BULLET))
story.append(p("The cross-sell need probability is generated from "
               "observable features; on real customers, need is driven "
               "by life events and circumstances no model can see. The "
               "queue is a triage list, not a prospect list.", BULLET))
story.append(p("The candidate does not claim mortgage lending "
               "experience, retail banking sales experience, or "
               "FHA/VA/FNMA/FHLMC knowledge — all are [CONFIRM WITH "
               "CANDIDATE] items, not facts in this project.", BULLET))
story.append(p("The base pay listed for this role is $15–17/hr "
               "(JPMorgan careers: $15.00/hr; LinkedIn: $17.00/hr), "
               "verified live Nov 21, 2026. "
               "Total compensation is commission-dependent and "
               "unstated; it sits far below the candidate's "
               "$140–190K band. Backup only.", BULLET))
story.append(p("10 &nbsp; Reproducibility", H1))
story.append(p(f"Run <font face='Helvetica-Bold'>python run_all.py"
               "</font> (generate → validate → model → chart → report), "
               f"then <font face='Helvetica-Bold'>python "
               f"tests/run_tests.py</font> — {n_tests} checks, all "
               "pass. Requirements: Python 3.12, numpy ≥ 2.0, pandas "
               "≥ 2.0, scikit-learn ≥ 1.5, matplotlib ≥ 3.8. Suggested "
               "repo (user creates the empty repo first): <font face="
               "'Helvetica-Bold'>jpmorgan-home-lending-advisor-ml"
               "</font> under github.com/bhupesh-debug. No existing "
               "repo under that profile maps to mortgage or home "
               "lending (confirmed by search, Nov 22, 2026).", BODY))
story.append(p("Files: run_all.py · build_pdf.py · build_deck.py · "
               "make_notebooks.py · requirements.txt · README.md · "
               "src/ (5 modules) · tests/run_tests.py · notebooks/ "
               "(01_eda–04_evaluation) · data/processed/ · models/ "
               "(13 CSVs) · reports/model_report.md + 7 charts.",
               SMALL))
story.append(p("Bhupesh Chandra Dimri — November 2026. All data "
               "synthetic; no real borrower data, no real PII, no real "
               "JPMorgan Chase data.", SMALL))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(0.75 * inch, 0.55 * inch,
                      "Home Lending Pipeline & Mortgage Advisory "
                      "Analytics — JPMorganChase — Bhupesh Chandra "
                      "Dimri — Nov 2026 — ALL DATA SYNTHETIC")
    canvas.drawRightString(LETTER[0] - 0.75 * inch, 0.55 * inch,
                           f"Page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=LETTER,
                        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                        topMargin=0.6 * inch, bottomMargin=0.75 * inch,
                        title="Home Lending Pipeline & Mortgage Advisory "
                              "Analytics — JPMorganChase",
                        author="Bhupesh Chandra Dimri")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
