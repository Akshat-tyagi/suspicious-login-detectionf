"""Machine-learning models: Isolation Forest and supervised classifiers.  (split from detector.py)"""

import numpy as np
import pandas as pd
from sld.config import CONFIG


# ---- ml ----
from sklearn.ensemble import IsolationForest

ML_FEATURES = ["new_ip", "new_device", "new_country", "hour_dev", "log_speed"]
ML_EXTRA = ["new_asn", "log_ip_fails"]        # used only when the data really carries ASN information


def add_ml_scores(feat, cfg=CONFIG, fit_mask=None):
    """Unsupervised Isolation Forest on SUCCESSFUL logins (account-takeover view).

    Rules already handle failure bursts well. The model's job is different: learn what a normal
    successful login looks like (known IP/device/network/country, usual hour, plausible travel,
    quiet source address) and flag rare combinations that no single rule covers. No labels are used.
    """
    f = feat.copy()
    cols = ML_FEATURES + (ML_EXTRA if "asn" in f and f["asn"].nunique() > 1 else [])
    ok = f["success"] == 1
    X = f.loc[ok, cols].astype(float)
    fit_rows = ok if fit_mask is None else ok & pd.Series(np.asarray(fit_mask), index=f.index)
    model = IsolationForest(n_estimators=300, contamination=cfg["ml_contamination"],
                            random_state=cfg["seed"]).fit(f.loc[fit_rows, cols].astype(float))
    raw = -model.score_samples(X)
    f["ml_anomaly"] = 0.0
    f.loc[ok, "ml_anomaly"] = 100 * (raw - raw.min()) / (raw.max() - raw.min())
    f["ml_flag"] = False
    f.loc[ok, "ml_flag"] = model.predict(X) == -1
    q = X.quantile(.97)                       # explanation: which features are unusually high?
    f["ml_reason"] = ""
    for i in X.index[f.loc[X.index, "ml_flag"]]:
        odd = [f"{c}={X.at[i, c]:.4g}" for c in cols if X.at[i, c] > q[c] and X.at[i, c] > 0]
        f.at[i, "ml_reason"] = ", ".join(odd) or "rare combination of features"
    return f, model


# ---- supervised ----
SUP_FEATURES = ["success", "fails_user_10m", "fails_ip_10m", "distinct_users_ip_10m", "fails_ip_24h",
                "distinct_users_ip_24h", "log_ip_fails", "prior_successes", "new_ip", "new_device",
                "new_country", "new_asn", "hour_dev", "log_speed", "countries_seen_before",
                "share_ip", "share_device", "share_asn", "share_country",
                "asn_fail_rate", "country_fail_rate", "ip_fail_rate", "user_fail_rate",
                "log_ip_prior", "log_asn_prior", "rtt_log", "rtt_dev", "device_code", "country_code"]
_SUP_CATEGORICAL = ("device_code", "country_code")


def _make_hgb(cfg, kind):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=200 if kind == "ip" else 150, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=20,
        l2_regularization=1.0, early_stopping=False, random_state=cfg["seed"],
        categorical_features=np.array([c in _SUP_CATEGORICAL for c in SUP_FEATURES]))


def add_supervised_scores(feat, cfg=CONFIG):
    """Two gradient-boosting models trained on the real labels, evaluated honestly:

      * attack-IP model  (all logins)          -> 'does this login look like known attacker traffic?'
      * takeover model   (successful logins)   -> 'does this login look like a confirmed account takeover?'

    Protocol (no leakage): users are split train/test (column 'split'). Train users get OUT-OF-FOLD scores
    (each fold is scored by a model that never saw those users); test users are scored by a model fitted on
    all train users. Thresholds are chosen on the train users' out-of-fold scores ONLY, then applied
    unchanged to test users. Returns (frame_with_scores, report_dataframe).
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
    f = feat.copy()
    X = f[SUP_FEATURES].astype(float).to_numpy()
    train = (f["split"] == "train").to_numpy()
    test = ~train
    ok = (f["success"] == 1).to_numpy()
    fold = pd.util.hash_pandas_object(f["user"].astype(str) + "#sup", index=False).to_numpy() % cfg["sup_folds"]
    jobs = {"ip": ((f["gt_attack_ip"] == 1).to_numpy(), np.ones(len(f), bool), "attack-IP model"),
            "ato": ((f["gt_label"] == 1).to_numpy(), ok, "takeover model")}
    report = []
    for key, (y, rows, name) in jobs.items():
        w = np.where(y, 30.0 if key == "ato" else 1.0, 1.0)          # takeover is extremely rare: up-weight
        score = np.zeros(len(f))
        for k in range(cfg["sup_folds"]):                             # out-of-fold scores for train users
            fit, hold = train & rows & (fold != k), train & rows & (fold == k)
            if y[fit].sum() == 0 or not hold.any():
                continue
            score[hold] = _make_hgb(cfg, key).fit(X[fit], y[fit], sample_weight=w[fit]).predict_proba(X[hold])[:, 1]
        tr, te = train & rows, test & rows
        final = _make_hgb(cfg, key).fit(X[tr], y[tr], sample_weight=w[tr])
        score[te] = final.predict_proba(X[te])[:, 1]
        if key == "ip":                                               # threshold = best F1 on train (out-of-fold)
            p, r, t = precision_recall_curve(y[tr], score[tr])
            f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
            thr = float(t[int(np.argmax(f1))])
        else:                                                         # threshold = flag the riskiest X% of logins
            thr = float(np.quantile(score[tr], 1 - cfg["sup_ato_budget"]))
        flag = (score >= thr) & rows
        f[f"sup_{key}_score"], f[f"sup_{key}_flag"] = score, flag
        for part, m in (("train (out-of-fold)", tr), ("test (held-out users)", te)):
            yy, ss = y[m], score[m]
            both = 0 < yy.sum() < len(yy)
            report.append({"model": name, "evaluated_on": part, "threshold": round(thr, 4),
                           "positives": int(yy.sum()), "flagged": int(flag[m].sum()),
                           "true_flagged": int((flag[m] & yy).sum()),
                           "roc_auc": float(roc_auc_score(yy, ss)) if both else np.nan,
                           "average_precision": float(average_precision_score(yy, ss)) if both else np.nan})
    return f, pd.DataFrame(report)
