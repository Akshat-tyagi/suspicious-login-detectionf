"""Rules, risk score, plain-English reasons, grouping alerts into incidents.  (split from detector.py)"""

import pandas as pd
from sld.config import CONFIG


# ---- scoring ----
MODEL_COLS = ["ML_anomaly", "SUP_attack_ip_model", "SUP_takeover_model"]
RULE_COLS =["R1_brute_force", "R2_password_spraying", "R3_success_after_failures",
             "R3b_unfamiliar_source", "R4_impossible_travel", "R5_new_device_odd_hour", "R6_slow_spray",
             "R7_new_network_and_device"]


def score_events(feat, cfg=CONFIG, use_ml=True, explain=True, use_sup=True):
    """Apply the explainable rules, add ML evidence, produce risk score, level and reasons."""
    f, p = feat.copy(), cfg["points"]
    failed, ok = f["success"] == 0, f["success"] == 1
    f["R1_brute_force"] = failed & (f["fails_user_10m"] >= cfg["brute_force_fails"])
    f["R2_password_spraying"] = failed & (f["distinct_users_ip_10m"] >= cfg["spray_users"])
    f["R3_success_after_failures"] = ok & (f["fails_user_10m"] >= cfg["success_after_fails"])
    unfamiliar = (f["new_ip"] + f["new_device"] + f["new_country"] + f["new_asn"]) > 0
    f["R3b_unfamiliar_source"] = f["R3_success_after_failures"] & unfamiliar
    f["R4_impossible_travel"] = ok & (f["travel_speed_kmh"] > cfg["max_speed_kmh"]) \
        & (f["countries_seen_before"] < cfg["travel_max_known_countries"])
    f["R5_new_device_odd_hour"] = ok & (f["new_device"] == 1) & (f["hour_dev"] >= cfg["odd_hour_dev"])
    f["R6_slow_spray"] = failed & (f["distinct_users_ip_24h"] >= cfg["slow_spray_users"]) \
        & (f["fails_ip_24h"] >= cfg["slow_spray_users"])
    # R7: a login from a network (ASN) AND a browser/OS/device combination this user never used before.
    # One new thing is normal (new phone, hotel wifi); both at once is how a stolen password looks.
    f["R7_new_network_and_device"] = ok & (f["new_asn"] == 1) & (f["new_device"] == 1)
    ml = f["ml_flag"] if (use_ml and "ml_flag" in f) else pd.Series(False, index=f.index)
    f["ML_anomaly"] = ml.astype(bool)
    no = pd.Series(False, index=f.index)
    f["SUP_attack_ip_model"] = (f["sup_ip_flag"] if (use_sup and "sup_ip_flag" in f) else no).astype(bool)
    f["SUP_takeover_model"] = (f["sup_ato_flag"] if (use_sup and "sup_ato_flag" in f) else no).astype(bool)

    score = (f["R1_brute_force"] * p["R1"] + (f["R1_brute_force"] & (f["fails_user_10m"] >= 10)) * p["R1_heavy"]
             + f["R2_password_spraying"] * p["R2"]
             + (f["R2_password_spraying"] & (f["distinct_users_ip_10m"] >= 8)) * p["R2_heavy"]
             + f["R3_success_after_failures"] * p["R3"] + f["R3b_unfamiliar_source"] * p["R3b"]
             + f["R4_impossible_travel"] * p["R4"] + f["R5_new_device_odd_hour"] * p["R5"]
             + f["R6_slow_spray"] * p["R6"] + f["R7_new_network_and_device"] * p["R7"]
             + f["ML_anomaly"] * p["ML"]
             + f["SUP_attack_ip_model"] * p.get("SUP_IP", 40) + f["SUP_takeover_model"] * p.get("SUP_ATO", 40))
    f["risk_score"] = score.clip(upper=100).astype(int)
    f["risk_level"] = pd.cut(f["risk_score"], [-1, 0, cfg["level_medium"] - 1, cfg["level_high"] - 1, 100],
                             labels=["None", "Low", "Medium", "High"]).astype(str)
    f["alert"] = f["risk_score"] >= cfg["alert_threshold"]

    def _explain(r):
        out, w = [], cfg["window_minutes"]
        if r.R1_brute_force:
            out.append(f"Brute force: {r.fails_user_10m} failed logins for {r.user} within {w} min")
        if r.R2_password_spraying:
            out.append(f"Password spraying: {r.ip} failed against {r.distinct_users_ip_10m} different accounts within {w} min")
        if r.R3_success_after_failures:
            out.append(f"Successful login after {r.fails_user_10m} recent failures")
        if r.R3b_unfamiliar_source:
            new = [n for n, v in (("IP", r.new_ip), ("device", r.new_device), ("country", r.new_country),
                                  ("network", r.new_asn)) if v]
            out.append("...from a source this user has not used before (new " + "/".join(new) + ")")
        if r.R4_impossible_travel:
            out.append(f"Impossible travel: login from {r.country} implies ~{r.travel_speed_kmh:,.0f} km/h since the previous login")
        if r.R5_new_device_odd_hour:
            out.append(f"New device used {r.hour_dev:.1f} h away from this user's usual login time")
        if r.R6_slow_spray:
            out.append(f"Slow password spraying: {r.ip} failed against {r.distinct_users_ip_24h} accounts in {cfg['long_window_hours']} h")
        if r.R7_new_network_and_device:
            out.append("Login from a network (ASN) and a browser/OS/device combination this user has never used before")
        if r.ML_anomaly:
            out.append(f"ML anomaly (Isolation Forest): {r.ml_reason}")
        if r.SUP_attack_ip_model:
            out.append(f"Supervised model: login resembles known attacker traffic (score {r.sup_ip_score:.2f})")
        if r.SUP_takeover_model:
            out.append(f"Supervised model: login resembles a confirmed account takeover (score {r.sup_ato_score:.2f})")
        return "; ".join(out)

    f["reasons"], f["rules"] = "", ""
    if not explain:                      # fast mode for threshold sweeps (scores only)
        return f
    hit = f.index[f["risk_score"] > 0]
    f.loc[hit, "reasons"] = [_explain(r) for r in f.loc[hit].itertuples()]
    names = RULE_COLS + MODEL_COLS
    f.loc[hit, "rules"] = f.loc[hit, names].apply(lambda r: ", ".join(c for c in names if r[c]), axis=1)
    return f


# ---- incidents ----
def build_incidents(scored, cfg=CONFIG):
    """Group related alerts (same IP or same user, close in time) into incidents so the
    administrator reviews ONE item per attack instead of dozens of raw alerts."""
    a = scored[scored["alert"]].sort_values("timestamp").copy()
    if a.empty:
        return pd.DataFrame(), a
    gap = pd.Timedelta(minutes=cfg["incident_gap_minutes"])
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    last = {"ip": {}, "user": {}}
    for i, r in zip(a.index, a.itertuples()):
        parent[i] = i
        for kind, key in (("ip", r.ip), ("user", r.user)):
            prev = last[kind].get(key)
            if prev is not None and r.timestamp - prev[1] <= gap:
                parent[find(i)] = find(prev[0])
            last[kind][key] = (i, r.timestamp)
    a["incident_id"] = pd.factorize(pd.Series([find(i) for i in a.index], index=a.index))[0] + 1

    rows = []
    for iid, g in a.groupby("incident_id"):
        top = g.sort_values("risk_score", ascending=False).iloc[0]
        rows.append({
            "incident_id": iid, "start": g["timestamp"].min(), "end": g["timestamp"].max(),
            "n_alerts": len(g), "max_score": int(g["risk_score"].max()),
            "risk_level": top["risk_level"],
            "users": ", ".join(sorted(set(g["user"]))[:cfg["incident_list_max"]]),
            "ips": ", ".join(sorted(set(g["ip"]))[:cfg["incident_list_max"]]),
            "n_users": g["user"].nunique(), "n_ips": g["ip"].nunique(),
            "rules": ", ".join(sorted(set(", ".join(g["rules"]).split(", ")))),
            "summary": top["reasons"],
        })
    return pd.DataFrame(rows), a
