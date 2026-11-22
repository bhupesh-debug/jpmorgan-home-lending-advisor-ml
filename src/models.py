"""Model training + evaluation for the three modules.

Module 1 — monthly origination forecasting (regression):
    seasonal-naive baseline (same month last year = lag12) |
    naive baseline (last month = lag1) | Linear Regression |
    Random Forest. Metrics: WAPE, MAPE, MAE, R^2 on the time
    hold-out (Jul-Dec 2025).
Module 2 — affordability band (3-class) and product fit (4-class):
    simple-rule baseline (DTI-only / simplified product rule) |
    Logistic Regression | Random Forest. Metrics: per-class
    precision/recall/F1, macro-F1, accuracy on the time hold-out
    (application_date >= 2025-07-01).
Module 3 — cross-sell need (binary): rule baseline | Logistic
    Regression | Random Forest. Metrics: precision, recall, F1,
    ROC-AUC on a stratified 80/20 hold-out. Review queue: top 20 by
    predicted probability.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler


def _reg_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # MAPE only over non-zero actuals: funded volume is exactly 0 in
    # some small branch-product-months (no loans funded), and dividing
    # by ~0 would explode the metric. WAPE is primary regardless.
    nz = np.abs(y_true) > 1e-6
    mape = float(np.mean(np.abs(y_true[nz] - y_pred[nz])
                         / np.abs(y_true[nz]))) if nz.any() else np.nan
    # WAPE is primary: funded volume is lumpy (spring peaks, rate
    # moves), so MAPE explodes on small-denominator branch-months.
    wape = float(np.sum(np.abs(y_true - y_pred))
                 / np.maximum(np.sum(np.abs(y_true)), 1e-9))
    return {"wape": wape,
            "mape": mape,
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred))}


def origination_models(X_train, y_train, X_test, y_test):
    """Baselines read predictions straight off the lag features."""
    rows = []
    rows.append({"model": "Seasonal-naive baseline (same month last year)",
                 **_reg_metrics(y_test, X_test["lag12"])})
    rows.append({"model": "Naive baseline (last month)",
                 **_reg_metrics(y_test, X_test["lag1"])})
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    rows.append({"model": "Linear Regression",
                 **_reg_metrics(y_test, lr.predict(X_test))})
    rf = RandomForestRegressor(n_estimators=300, random_state=42,
                               n_jobs=-1)
    rf.fit(X_train, y_train)
    rows.append({"model": "Random Forest",
                 **_reg_metrics(y_test, rf.predict(X_test))})
    imp = pd.Series(rf.feature_importances_,
                    index=X_train.columns).sort_values(ascending=False)
    return rows, imp, rf


def _clf_metrics(y_true, y_pred, classes):
    out = {}
    for cls in classes:
        yt = (y_true == cls).astype(int)
        yp = (y_pred == cls).astype(int)
        out[f"p_{cls}"] = float(precision_score(yt, yp, zero_division=0))
        out[f"r_{cls}"] = float(recall_score(yt, yp, zero_division=0))
        out[f"f1_{cls}"] = float(f1_score(yt, yp, zero_division=0))
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro",
                                     zero_division=0))
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    return out


def band_models(X_train, y_train, X_test, y_test, rule_pred):
    classes = ["Affordable", "Stretch", "Unaffordable"]
    rows = []
    rows.append({"model": "DTI-only rule baseline",
                 **_clf_metrics(y_test, rule_pred, classes)})
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xte_s = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                            random_state=42)
    lr.fit(Xtr_s, y_train)
    rows.append({"model": "Logistic Regression",
                 **_clf_metrics(y_test, lr.predict(Xte_s), classes)})
    rf = RandomForestClassifier(n_estimators=300,
                                class_weight="balanced_subsample",
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rows.append({"model": "Random Forest",
                 **_clf_metrics(y_test, rf.predict(X_test), classes)})
    imp = pd.Series(rf.feature_importances_,
                    index=X_train.columns).sort_values(ascending=False)
    return rows, imp, rf


def fit_models(X_train, y_train, X_test, y_test, rule_pred):
    classes = ["Conventional", "FHA", "VA", "Jumbo"]
    rows = []
    rows.append({"model": "Simplified product rule baseline",
                 **_clf_metrics(y_test, rule_pred, classes)})
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xte_s = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                            random_state=42)
    lr.fit(Xtr_s, y_train)
    rows.append({"model": "Logistic Regression",
                 **_clf_metrics(y_test, lr.predict(Xte_s), classes)})
    rf = RandomForestClassifier(n_estimators=300,
                                class_weight="balanced_subsample",
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rows.append({"model": "Random Forest",
                 **_clf_metrics(y_test, rf.predict(X_test), classes)})
    imp = pd.Series(rf.feature_importances_,
                    index=X_train.columns).sort_values(ascending=False)
    return rows, imp, rf


def _bin_metrics(y_true, y_pred, y_proba):
    return {
        "precision": float(precision_score(y_true, y_pred,
                                           zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def crosssell_models(X_train, y_train, X_test, y_test, rule_pred):
    rows = []
    rows.append({"model": "Rule baseline (no mortgage + tenure + event)",
                 **_bin_metrics(y_test, rule_pred,
                                rule_pred.astype(float))})
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xte_s = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                            random_state=42)
    lr.fit(Xtr_s, y_train)
    proba_lr = lr.predict_proba(Xte_s)[:, 1]
    rows.append({"model": "Logistic Regression",
                 **_bin_metrics(y_test, lr.predict(Xte_s), proba_lr)})
    rf = RandomForestClassifier(n_estimators=300,
                                class_weight="balanced_subsample",
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    proba_rf = rf.predict_proba(X_test)[:, 1]
    rows.append({"model": "Random Forest",
                 **_bin_metrics(y_test, rf.predict(X_test), proba_rf)})
    imp = pd.Series(rf.feature_importances_,
                    index=X_train.columns).sort_values(ascending=False)
    return rows, imp, rf, proba_rf


def review_queue_metrics(y_true, proba, top_n=20):
    """Top-N ranked outreach list: how many of the top N are truly
    mortgage-need customers, and what share of ALL need customers the
    top N captures."""
    order = np.argsort(-np.asarray(proba))
    top = order[:top_n]
    y = np.asarray(y_true)
    n_need_top = int(y[top].sum())
    n_need_all = int(y.sum())
    return {
        "queue_size": top_n,
        "need_in_top": n_need_top,
        "need_total_holdout": n_need_all,
        "precision_at_n": n_need_top / top_n,
        "recall_of_need": (n_need_top / n_need_all if n_need_all
                            else 0.0),
        "base_rate": float(y.mean()),
    }
