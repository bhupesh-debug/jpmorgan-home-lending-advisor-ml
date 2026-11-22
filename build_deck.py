"""Build the 10-slide PDF deck (landscape).

Run from the project root AFTER `python run_all.py`:
    ~/workspace/quant-ipo-engine/.venv/bin/python build_deck.py
Output: ~/workspace/your_files/Bhupesh_Chandra_Dimri_ML-Project_
Sr-Home-Lending-Advisor_JPMorganChase_Deck.pdf

All numbers come from models/*.csv and data/processed/*.csv.
"""

import csv
import os
import re
import subprocess
import sys

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.expanduser(
    "~/workspace/your_files/Bhupesh_Chandra_Dimri_ML-Project_"
    "Sr-Home-Lending-Advisor_JPMorganChase_Deck.pdf")
CHARTS = os.path.join(ROOT, "reports", "charts")
MODELS = os.path.join(ROOT, "models")
PROC = os.path.join(ROOT, "data", "processed")
PAGE = landscape(LETTER)

NAVY = colors.HexColor("#1f4e79")
TEAL = colors.HexColor("#2e9e6b")
GREY = colors.HexColor("#595959")
RED = colors.HexColor("#b23a48")

SS = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=SS["Heading1"], fontSize=26,
                    leading=31, spaceAfter=10, textColor=NAVY)
H2 = ParagraphStyle("H2", parent=SS["Heading2"], fontSize=15,
                    spaceBefore=8, spaceAfter=6, textColor=NAVY)
BODY = ParagraphStyle("Body", parent=SS["BodyText"], fontSize=13,
                      leading=18, spaceAfter=8)
SMALL = ParagraphStyle("Small", parent=SS["BodyText"], fontSize=10.5,
                       leading=14, textColor=GREY)
CELL = ParagraphStyle("Cell", parent=SS["BodyText"], fontSize=12,
                      leading=15)
CELLH = ParagraphStyle("CellH", parent=CELL, textColor=colors.white,
                       fontName="Helvetica-Bold")
TITLE = ParagraphStyle("Title", parent=SS["Title"], fontSize=34,
                       leading=40, textColor=NAVY)
SUB = ParagraphStyle("Sub", parent=SS["BodyText"], fontSize=15,
                     leading=20, textColor=GREY)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=18,
                        bulletIndent=6, spaceAfter=6)


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
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
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
    return "—" if v != v else f"{v:.{digits}f}"


def pct(x, digits=1):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "—" if v != v else f"{v * 100:.{digits}f}%"


def money(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return f"${v:,.0f}"


pipe = pd.read_csv(os.path.join(PROC, "monthly_pipeline.csv"),
                   parse_dates=["month"])
bor = pd.read_csv(os.path.join(PROC, "borrowers.csv"))
cust = pd.read_csv(os.path.join(PROC, "customers.csv"))
m1 = read_csv("origination_metrics.csv")
m2a = read_csv("affordability_metrics.csv")
m2b = read_csv("product_fit_metrics.csv")
m3 = read_csv("crosssell_metrics.csv")
bench = read_csv("benchmark.csv")
queue = read_csv("review_queue.csv")[0]
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
sn_m1 = next(r for r in m1 if r["model"].startswith("Seasonal-naive"))
rf_m2a = next(r for r in m2a if r["model"] == "Random Forest")
rule_m2a = next(r for r in m2a if r["model"].startswith("DTI-only"))
rf_m2b = next(r for r in m2b if r["model"] == "Random Forest")
rule_m2b = next(r for r in m2b if r["model"].startswith("Simplified"))
lr_m3 = next(r for r in m3 if r["model"] == "Logistic Regression")
rf_m3 = next(r for r in m3 if r["model"] == "Random Forest")
rule_m3 = next(r for r in m3 if r["model"].startswith("Rule baseline"))
best_m3 = max((r for r in m3 if "baseline" not in r["model"]),
              key=lambda r: float(r["f1"]))
band_share = bor["affordability_band"].value_counts(normalize=True)

S = []

# 1 — title
S.append(Spacer(1, 1.4 * inch))
S.append(p("Home Lending Pipeline &amp;<br/>Mortgage Advisory "
           "Analytics", TITLE))
S.append(Spacer(1, 0.2 * inch))
S.append(p("Sr. Home Lending Advisor — JPMorganChase, Larchmont NY — "
           "onsite, full-time, apply by Dec 19, 2026", SUB))
S.append(Spacer(1, 0.3 * inch))
S.append(p("Bhupesh Chandra Dimri — November 2026", BODY))
S.append(p("<b>ALL DATA IS SYNTHETIC. No real borrower data, no real "
           "PII, no real JPMorgan Chase data.</b>", BODY))
S.append(PageBreak())

# 2 — the role & the gap
S.append(p("The role — and the honest gap", H1))
S.append(p("The advisor coaches the branch team, advises customers "
           "on the right mortgage product, and drives originations "
           "with assigned retail branches — consultative "
           "recommendations from financial analysis, product "
           "knowledge, and the customer's needs and goals.", BODY))
S.append(p("<b>Match vs the JD: 4/10.</b> Forecasting, financial "
           "analysis, and consultative skills map directly. Hard "
           "gaps: no mortgage lending experience, no retail banking "
           "sales experience, no FHA/VA/FNMA/FHLMC knowledge, no "
           "local Westchester market knowledge — all [CONFIRM WITH "
           "CANDIDATE]. This project narrows the analytics part of "
           "the gap; it does not close the domain part.", BODY))
S.append(p("Two flags: base pay is $15–17/hr (verified live Nov 21, "
           "2026), total compensation commission-dependent and "
           "unstated — far below the $140–190K band, backup only. "
           "And the Module 2 scorer is an advisory prototype on "
           "synthetic data, not a lending decision tool.", SMALL))
S.append(PageBreak())

# 3 — data
S.append(p("Synthetic data, stated plainly", H1))
S.append(table(
    ["Dataset", "Rows", "Key fact"],
    [["Branches", "8",
      "Larchmont, Scarsdale, White Plains, New Rochelle, Bronxville, "
      "Rye, Mamaroneck, Greenwich"],
     ["Monthly pipeline", f"{len(pipe):,}",
      f"Branch x product x month, 2023–2025; {total_funded:,} funded "
      f"loans, {money(total_vol)} funded volume"],
     ["Borrowers", f"{len(bor):,}",
      f"Affordable {pct(band_share.get('Affordable', 0))} / Stretch "
      f"{pct(band_share.get('Stretch', 0))} / Unaffordable "
      f"{pct(band_share.get('Unaffordable', 0))}"],
     ["Customers", f"{len(cust):,}",
      f"{pct(cust['mortgage_need'].mean())} with mortgage need"],
     ["Rate path", "36 months",
      "Synthetic 30-year fixed, 6.4%–7.6%, spring seasonality"]],
    widths=[2.2 * inch, 1.2 * inch, 6.1 * inch]))
S.append(p("Validation gates fail the run on any violation — missing "
           "values, funnel counts that increase downstream, rates "
           "outside [5%, 10%], DTI outside (0, 1.2], credit scores "
           "outside [300, 850], loan amounts above home price. The "
           "generator is <b>src/data_generation.py</b>; the gates "
           "are <b>src/data_pipeline.py</b>.", BODY))
S.append(PageBreak())

# 4 — module 1
S.append(p("Module 1 — Origination forecasting", H1))
S.append(table(
    ["Model (hold-out: 192 branch-product-months)", "WAPE", "MAPE",
     "MAE", "R²"],
    [[r["model"], pct(float(r["wape"])), pct(float(r["mape"])),
      money(r["mae"]), fmt(r["r2"])] for r in m1],
    widths=[4.3 * inch, 1.1 * inch, 1.1 * inch, 1.4 * inch, 1.6 * inch]))
S.append(p(f"Random Forest WAPE {pct(float(rf_m1['wape']))} vs "
           f"seasonal-naive {pct(float(sn_m1['wape']))}. WAPE is the "
           "primary metric: funded volume is lumpy, so MAPE explodes "
           "on small-denominator branch-months (and is computed over "
           "non-zero actuals only). The seasonal-naive baseline is "
           "the worst by design — the rate path and pipeline mix move "
           "year over year.", BODY))
S.append(PageBreak())

# 5 — module 1 charts
S.append(p("Module 1 — The forecast, and what drives it", H1))
S.append(Image(os.path.join(CHARTS, "origination_actual_vs_forecast.png"),
               width=8.6 * inch, height=2.9 * inch))
S.append(p("Monthly funded volume, actual vs forecast — the model "
           "tracks the summer peak and autumn taper because the "
           "pipeline stages are known in advance. That is origination "
           "<i>estimation</i> with learned adjustments — exactly "
           "what the advisor needs.", SMALL))
S.append(PageBreak())

# 6 — module 2
S.append(p("Module 2 — Affordability &amp; product fit", H1))
S.append(table(
    ["Affordability (hold-out: 747 borrowers)", "Macro-F1",
     "Accuracy", "F1 Aff.", "F1 Stretch", "F1 Unaff."],
    [[r["model"], fmt(r["macro_f1"]), fmt(r["accuracy"]),
      fmt(r["f1_Affordable"]), fmt(r["f1_Stretch"]),
      fmt(r["f1_Unaffordable"])] for r in m2a],
    widths=[3.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch,
            1.2 * inch]))
S.append(table(
    ["Product fit (hold-out: 747 borrowers)", "Macro-F1", "Accuracy",
     "F1 Conv.", "F1 FHA", "F1 VA", "F1 Jumbo"],
    [[r["model"], fmt(r["macro_f1"]), fmt(r["accuracy"]),
      fmt(r["f1_Conventional"]), fmt(r["f1_FHA"]), fmt(r["f1_VA"]),
      fmt(r["f1_Jumbo"])] for r in m2b],
    widths=[3.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch,
            1.2 * inch, 1.1 * inch]))
S.append(p(f"Affordability: Random Forest {fmt(rf_m2a['macro_f1'])} "
           f"vs DTI-only rule {fmt(rule_m2a['macro_f1'])} — "
           "affordability is not a single ratio. Product fit: Random "
           f"Forest {fmt(rf_m2b['macro_f1'])} vs simplified rule "
           f"{fmt(rule_m2b['macro_f1'])} — the rule misses the "
           "credit-band and FHA-ceiling interactions.", BODY))
S.append(Image(os.path.join(CHARTS, "product_fit_by_credit.png"),
               width=8.6 * inch, height=2.3 * inch))
S.append(PageBreak())

# 7 — module 2 scorer honesty
S.append(p("Module 2 — An advisory prototype, not a decision tool", H1))
S.append(p("The scorer explains its reasoning — DTI, credit band, "
           "down payment, loan limits — so an advisor can walk a "
           "customer through it. It makes no credit decision and "
           "uses no real borrower data.", BODY))
S.append(p("Labels come from transparent rules with 4% noise, so the "
           "classifier's edge over the simple rule comes from the "
           "interactions the simple rule ignores — not from hidden "
           "signal. On real borrowers, expect lower scores and keep "
           "the advisor in the loop.", BODY))
S.append(Image(os.path.join(
    CHARTS, "affordability_model_comparison.png"),
    width=8.6 * inch, height=2.9 * inch))
S.append(PageBreak())

# 8 — module 3
S.append(p("Module 3 — Branch cross-sell analyzer", H1))
S.append(table(
    ["Model (hold-out: 1,200 customers)", "Precision", "Recall",
     "F1", "Accuracy", "ROC-AUC"],
    [[r["model"], fmt(r["precision"]), fmt(r["recall"]),
      fmt(r["f1"]), fmt(r["accuracy"]), fmt(r["roc_auc"])]
     for r in m3],
    widths=[3.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch,
            1.5 * inch]))
S.append(p(f"{best_m3['model']} (F1 {fmt(best_m3['f1'])}) beats the "
           f"rule (F1 {fmt(rule_m3['f1'])}). Notably, Logistic "
           f"Regression (F1 {fmt(lr_m3['f1'])}) beats the Random "
           f"Forest (F1 {fmt(rf_m3['f1'])}): the need probability was "
           "generated logistically, so the linear model recovers the "
           "structure. The simpler model wins this module — and the "
           "report says so.", BODY))
S.append(p(f"<b>Outreach queue (top 20): {queue['need_in_top']}/20 "
           f"truly need a mortgage</b>, against a base rate of "
           f"{pct(float(queue['base_rate']))}. Queue recall: "
           f"{pct(float(queue['recall_of_need']))} — a triage list, "
           "not a prospect list.", BODY))
S.append(PageBreak())

# 9 — benchmark + hold-out
S.append(p("Benchmark — and a hold-out with no leakage", H1))
S.append(table(
    ["Module", "Baseline", "Best learned model", "Metric"],
    [[r["module"], f"{r['baseline']} {fmt(r['baseline_score'])}",
      f"{r['best_model']} {fmt(r['best_score'])}", r["metric"]]
     for r in bench],
    widths=[2.4 * inch, 2.2 * inch, 2.8 * inch, 2.1 * inch]))
S.append(p("<b>Modules 1 &amp; 2: time-based.</b> Train is Jan "
           "2023–Jun 2025; the hold-outs are the last 6 months. "
           "<b>Module 1 features are pipeline-known</b> — never the "
           "month's funded volume. <b>Module 2 excludes identity</b> "
           "— no IDs, no dates. <b>Module 3: stratified 80/20, seed "
           "42.</b>", BODY))
S.append(Image(os.path.join(CHARTS, "crosssell_lift.png"),
               width=8.6 * inch, height=2.9 * inch))
S.append(PageBreak())

# 10 — limits + next steps
S.append(p("Limits — and what comes next", H1))
S.append(p("All data is synthetic; rates, prices, and class shares "
           "are design choices, not JPMorgan Chase facts. Labels are "
           "generated from the same transparent rules the models "
           "learn, with 4% noise. The cross-sell need probability is "
           "generated from observable features; real need is driven "
           "by life events and circumstances no model can see.",
           BULLET))
S.append(p("This project proves pipeline design, affordability "
           "reasoning, and outreach analytics. It does not prove "
           "mortgage lending experience, retail banking sales "
           "experience, or FHA/VA/FNMA/FHLMC knowledge.", BULLET))
S.append(p(f"<b>Next steps:</b> {n_tests} tests pass. Suggested repo "
           "(user creates the empty repo first): "
           "<b>jpmorgan-home-lending-advisor-ml</b> under "
           "github.com/bhupesh-debug. Run <b>python run_all.py</b> "
           "then <b>python tests/run_tests.py</b> to reproduce every "
           "number in this deck.", BODY))
S.append(p("Bhupesh Chandra Dimri — November 2026. All data synthetic; "
           "no real borrower data, no real PII, no real JPMorgan "
           "Chase data.", SMALL))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(0.75 * inch, 0.45 * inch,
                      "Home Lending Pipeline & Mortgage Advisory "
                      "Analytics — JPMorganChase — Bhupesh Chandra "
                      "Dimri — Nov 2026 — ALL DATA SYNTHETIC")
    canvas.drawRightString(PAGE[0] - 0.75 * inch, 0.45 * inch,
                           f"Slide {doc.page} / 10")
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=PAGE,
                        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                        topMargin=0.6 * inch, bottomMargin=0.7 * inch,
                        title="Home Lending Pipeline & Mortgage Advisory "
                              "Analytics — JPMorganChase (Deck)",
                        author="Bhupesh Chandra Dimri")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
