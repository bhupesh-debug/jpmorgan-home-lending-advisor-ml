"""Charts for the report and deck. All charts are matplotlib, saved
as PNG under reports/charts/. Palette matches the PDF: navy #1f4e79,
teal #2e9e6b, grey, red for the negative case."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAVY = "#1f4e79"
TEAL = "#2e9e6b"
GREY = "#595959"
RED = "#b23a48"
LIGHT = "#dce6f1"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GREY,
                     "axes.labelcolor": GREY, "xtick.color": GREY,
                     "ytick.color": GREY})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def originations_by_product(pipe_path, out):
    pipe = pd.read_csv(pipe_path, parse_dates=["month"])
    piv = (pipe.groupby(["month", "product"])["funded_volume"]
           .sum().unstack(fill_value=0))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for prod in ["Conventional", "FHA", "VA", "Jumbo"]:
        ax.plot(piv.index, piv[prod] / 1e6, label=prod, linewidth=1.6)
    ax.set_ylabel("Funded volume ($M / month)")
    ax.set_title("Monthly funded volume by product, 2023-2025",
                 color=NAVY, fontsize=12)
    ax.legend(frameon=False, ncol=4)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def origination_actual_vs_forecast(pipe_path, holdout_path, out):
    pipe = pd.read_csv(pipe_path, parse_dates=["month"])
    hold = pd.read_csv(holdout_path, parse_dates=["month"])
    monthly = pipe.groupby("month")["funded_volume"].sum() / 1e6
    fc = hold.groupby("month")["y_pred_rf"].sum() / 1e6
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(monthly.index, monthly.values, label="Actual",
            color=NAVY, linewidth=1.8)
    ax.plot(fc.index, fc.values, label="Random Forest forecast",
            color=TEAL, linewidth=1.8, linestyle="--")
    ax.axvspan(pd.Timestamp("2025-07-01"), monthly.index.max(),
               color=LIGHT, alpha=0.6)
    ax.text(pd.Timestamp("2025-08-15"), monthly.max() * 0.92,
            "hold-out", color=GREY, fontsize=9)
    ax.set_ylabel("Funded volume ($M / month)")
    ax.set_title("Monthly funded volume: actual vs forecast",
                 color=NAVY, fontsize=12)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def feature_importance(imp_path, out, title):
    imp = pd.read_csv(imp_path).head(10)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.barh(imp["feature"][::-1], imp["importance"][::-1],
            color=NAVY)
    ax.set_xlabel("Random Forest feature importance")
    ax.set_title(title, color=NAVY, fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    _save(fig, out)


def model_comparison(metrics_path, metric_col, out, title):
    rows = pd.read_csv(metrics_path)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(rows["model"], rows[metric_col], color=[GREY, GREY, NAVY,
                                                   TEAL][:len(rows)])
    ax.set_ylabel(metric_col.replace("_", " ").title())
    ax.set_title(title, color=NAVY, fontsize=12)
    ax.tick_params(axis="x", labelrotation=18)
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(rows[metric_col]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    _save(fig, out)


def product_fit_by_credit(bor_path, out):
    bor = pd.read_csv(bor_path)
    tab = pd.crosstab(bor["credit_band"], bor["product_fit"],
                      normalize="index")
    order = ["Excellent", "Good", "Fair", "Poor"]
    tab = tab.reindex(order)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    bottom = np.zeros(len(tab))
    colors = [NAVY, TEAL, "#e8a838", RED]
    for j, prod in enumerate(["Conventional", "FHA", "VA", "Jumbo"]):
        vals = tab[prod].values
        ax.bar(tab.index, vals, bottom=bottom, label=prod,
               color=colors[j])
        bottom += vals
    ax.set_ylabel("Share of applications")
    ax.set_title("Product fit by credit band (rule-generated labels)",
                 color=NAVY, fontsize=12)
    ax.legend(frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, 1.22))
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def crosssell_lift(cust_path, out):
    cust = pd.read_csv(cust_path)
    tab = (cust.groupby("life_event")["mortgage_need"]
           .agg(["mean", "size"]).reset_index()
           .sort_values("mean", ascending=False))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(tab["life_event"], tab["mean"], color=NAVY)
    ax.set_ylabel("Share with mortgage need")
    ax.set_title("Mortgage need by life event (existing customers)",
                 color=NAVY, fontsize=12)
    ax.tick_params(axis="x", labelrotation=18)
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(tab["mean"]):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=9)
    _save(fig, out)


def correlation_heatmap(scored_path, out):
    cols = ["annual_income", "monthly_debts", "credit_score",
            "down_payment_pct", "home_price", "loan_amount",
            "front_dti", "back_dti", "reserves_months"]
    df = pd.read_csv(scored_path)
    corr = df[[c for c in cols if c in df.columns]].corr()
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right",
                       fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title("Borrower features, correlation (hold-out)",
                 color=NAVY, fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, out)
