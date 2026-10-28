"""Precision / recall / false-positive evaluation and threshold sweeps.  (split from detector.py)"""

import numpy as np
import pandas as pd
from sld.config import CONFIG
from sld.scoring import MODEL_COLS, RULE_COLS, score_events


# ---- evaluation ----
def confusion_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp, fp = int((y_true & y_pred).sum()), int((~y_true & y_pred).sum())
    fn, tn = int((y_true & ~y_pred).sum()), int((~y_true & ~y_pred).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "precision": prec, "recall": rec,
            "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def evaluate_events(feat, cfg=CONFIG):
    """Event-level comparison of the detection strategies against the ground-truth labels."""
    y = feat["gt_label"] == 1
    rules = score_events(feat, cfg, use_ml=False)
    hybrid = score_events(feat, cfg)
    rows = {
        "Rules only (alerts: Medium+)": rules["alert"],
        "ML only (Isolation Forest)": feat["ml_flag"],
        "Hybrid (alerts: Medium+)": hybrid["alert"],
        "Hybrid incl. Low watchlist": hybrid["risk_score"] >= 1,
    }
    return pd.DataFrame({k: confusion_metrics(y, v) for k, v in rows.items()}).T


def evaluate_incidents(scored, incidents, alerts):
    """Incident-level view: was each real attack noticed, how fast, and how many alerts were noise?"""
    gt = scored[scored["gt_attack_id"] > 0]
    alerted = set(scored.loc[scored["alert"], "event_id"])
    watch = set(scored.loc[scored["risk_score"] >= 1, "event_id"])
    rows = []
    for aid, g in gt.groupby("gt_attack_id"):
        hits = g[g["event_id"].isin(alerted)]
        delay = (hits["timestamp"].min() - g["timestamp"].min()).total_seconds() if len(hits) else np.nan
        rows.append({"attack_id": aid, "type": g["gt_attack_type"].iloc[0], "events": len(g),
                     "detected": len(hits) > 0, "on_watchlist": g["event_id"].isin(watch).any(),
                     "seconds_to_first_alert": delay})
    per_attack = pd.DataFrame(rows)
    by_type = per_attack.groupby("type").agg(attacks=("detected", "size"), detected=("detected", "sum"),
                                             on_watchlist=("on_watchlist", "sum"),
                                             median_seconds_to_alert=("seconds_to_first_alert", "median"))
    by_type["detection_rate"] = by_type["detected"] / by_type["attacks"]
    by_type["detection_or_watchlist_rate"] = by_type["on_watchlist"] / by_type["attacks"]
    if len(incidents):
        real = alerts.groupby("incident_id")["gt_label"].max()
        n_inc, n_real = len(real), int((real == 1).sum())
    else:
        n_inc = n_real = 0
    summary = {"attacks_total": len(per_attack), "attacks_detected": int(per_attack["detected"].sum()),
               "attack_detection_rate": per_attack["detected"].mean(),
               "attack_detection_rate_incl_watchlist": per_attack["on_watchlist"].mean(),
               "incidents_raised": n_inc, "incidents_that_are_real_attacks": n_real,
               "incident_precision": n_real / n_inc if n_inc else 0.0,
               "raw_alerts": int(len(alerts)), "alerts_per_incident": len(alerts) / n_inc if n_inc else 0.0}
    return summary, by_type, per_attack


# ---- RBA evaluation ----
def _split_by_user(users):
    """Deterministic 50/50 split by user: 'train' (fit the ML model, run threshold sweeps) vs
    'test' (final numbers). A user's logins never straddle both halves."""
    h = pd.util.hash_pandas_object(pd.Series(users).astype(str), index=False).to_numpy() % 2
    return np.where(h == 0, "train", "test")


def evaluate_rba_events(scored, cfg=CONFIG, mask=None):
    """Evaluate suspicious-login strategies against confirmed takeover labels only.

    Alerts without takeover labels may still represent other suspicious activity;
    these metrics describe takeover detection, not every type of suspicious login.
    """
    m = np.ones(len(scored), bool) if mask is None else np.asarray(mask, bool)
    rules = score_events(scored, cfg, use_ml=False, use_sup=False, explain=False)
    full = score_events(scored, cfg, explain=False)
    zero = np.zeros(len(scored), bool)
    sup_ip = scored["sup_ip_flag"].to_numpy(bool) if "sup_ip_flag" in scored else zero
    sup_ato = scored["sup_ato_flag"].to_numpy(bool) if "sup_ato_flag" in scored else zero
    strategies = {
        "Rules only (alerts: Medium+)": rules["alert"].to_numpy(),
        "Isolation Forest only": scored["ml_flag"].to_numpy(),
        "Supervised only (either model)": sup_ip | sup_ato,
        "Full hybrid: rules+ML+supervised": full["alert"].to_numpy(),
    }
    rows = []
    y = (scored["gt_label"] == 1).to_numpy()[m]
    for name, pred in strategies.items():
        rows.append({"label": "Account takeover", "strategy": name, "alerts": int(pred[m].sum()),
                     **confusion_metrics(y, pred[m])})
    return pd.DataFrame(rows)


def evaluate_rba_rules(scored, mask=None):
    """Contribution of combined rules and each model, counting each event once per row."""
    m = np.ones(len(scored), bool) if mask is None else np.asarray(mask, bool)
    atk, ato = (scored["gt_attack_ip"] == 1).to_numpy() & m, (scored["gt_label"] == 1).to_numpy() & m
    rows = []
    signals = {
        "Rules (combined)": scored[RULE_COLS].any(axis=1).to_numpy(bool),
        "Isolation Forest": scored["ML_anomaly"].to_numpy(bool),
        "Attack-IP model": scored["SUP_attack_ip_model"].to_numpy(bool),
        "Takeover model": scored["SUP_takeover_model"].to_numpy(bool),
    }
    for name, signal in signals.items():
        fired = signal & m
        n = int(fired.sum())
        rows.append({"model": name, "events_fired": n,
                     "precision_vs_attack_ip": (fired & atk).sum() / n if n else np.nan,
                     "recall_vs_attack_ip": (fired & atk).sum() / atk.sum() if atk.sum() else np.nan,
                     "takeover_events_hit": int((fired & ato).sum())})
    return pd.DataFrame(rows)


def evaluate_rba_summary(scored, alerts, mask=None):
    """Summarize takeover-event coverage for the selected evaluation users."""
    m = np.ones(len(scored), bool) if mask is None else np.asarray(mask, bool)
    ato = (scored["gt_label"] == 1).to_numpy() & m
    alert = scored["alert"].to_numpy()
    total = int(ato.sum())
    detected = int((ato & alert).sum())
    out = {
        "Actual takeover events": str(total),
        "Detected takeover events": str(detected),
        "Missed takeover events": str(total - detected),
        "Takeover detection rate": f"{detected / total * 100:.2f}%" if total else "N/A",
    }
    return pd.DataFrame({"metric": list(out), "value": list(out.values())})


def sensitivity_sweep(scored, cfg=CONFIG, mask=None):
    """Vary one threshold at a time (on the TRAIN users) and show the trade-off."""
    m = np.ones(len(scored), bool) if mask is None else np.asarray(mask, bool)
    atk, ato = (scored["gt_attack_ip"] == 1).to_numpy() & m, (scored["gt_label"] == 1).to_numpy() & m
    grid = {"brute_force_fails": [3, 5, 8, 12], "spray_users": [3, 4, 6, 10], "max_speed_kmh": [900, 2000, 5000]}
    rows = []
    for param, values in grid.items():
        for v in values:
            a = score_events(scored, {**cfg, param: v}, explain=False)["alert"].to_numpy() & m
            tp = int((a & atk).sum())
            rows.append({"parameter": param, "value": v, "alerts": int(a.sum()),
                         "precision_vs_attack_ip": tp / a.sum() if a.sum() else np.nan,
                         "recall_vs_attack_ip": tp / atk.sum() if atk.sum() else np.nan,
                         "takeover_events_alerted": int((a & ato).sum()), "takeover_events": int(ato.sum())})
    return pd.DataFrame(rows)
