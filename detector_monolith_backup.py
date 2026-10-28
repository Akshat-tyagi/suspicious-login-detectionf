"""
Python-Based Suspicious Login Detection System
==============================================
Pipeline:  generate/read logs -> validate & clean -> feature engineering
           -> explainable rules + Isolation Forest -> risk score & reasons
           -> incidents -> SQLite storage -> evaluation.

Final project (Milestone 4): the same pipeline now runs on the public RBA login dataset
(Wiefling et al., 2022, CC BY 4.0, doi 10.1145/3546069) instead of simulated logs.

    python detector.py                      # RBA run (reads data/rba-dataset.csv)
    python detector.py --rows 300000        # smaller / faster subset
    python detector.py --demo               # old Milestone-3 simulated data
    streamlit run app.py                    # dashboard

Honest scope: RBA is synthesised from real login behaviour. Countries/IPs are artificial,
so geography-based rules are expected to be weaker on it than on genuine logs. The device
fingerprint (browser + OS + device type) and the network (ASN) are the stronger signals.
"""

# ==== SECTION: config ====
import hashlib
import ipaddress
import json
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

CONFIG = {
    "seed": 42,
    # --- time windows ---
    "window_minutes": 10,          # short window for burst detection
    "long_window_hours": 24,       # long window for slow / low-volume attacks
    "incident_gap_minutes": 30,    # alerts closer than this are merged into one incident
    # --- rule thresholds (adjustable starting points, not proof of an attack) ---
    "brute_force_fails": 5,        # R1: failed logins for ONE account in the window
    "spray_users": 4,              # R2: ONE IP failing against this many accounts in the window
    "success_after_fails": 3,      # R3: successful login right after this many failures
    "max_speed_kmh": 900,          # R4: faster than a passenger jet = impossible travel
    "odd_hour_dev": 6.0,           # R5: hours away from the user's usual login time
    "slow_spray_users": 8,         # R6: distinct accounts failed from one IP in 24h
    "min_history": 5,              # successful logins needed before "new X" is meaningful
    # --- risk scoring ---
    "points": {"R1": 50, "R1_heavy": 15, "R2": 50, "R2_heavy": 15, "R3": 20, "R3b": 25,
               "R4": 60, "R5": 20, "R6": 45, "R7": 35, "ML": 25,   # R5 weakened: alone it is below base rate on RBA
               "SUP_IP": 40, "SUP_ATO": 40},                      # supervised models (RBA only)
    "sup_folds": 3,                    # cross-fitting folds (by user) so every score is out-of-sample
    "sup_ato_budget": 0.01,            # takeover model flags the riskiest 1% of successful logins
    "travel_max_known_countries": 3,   # R4 is skipped for 'frequent hoppers' (VPN users, frequent flyers)
    # ML alone (25) stays Low = "watchlist"; it becomes an alert only together with another signal.
    "level_medium": 30,            # score >= 30 -> Medium  (this is also the alert threshold)
    "level_high": 60,              # score >= 60 -> High
    "alert_threshold": 30,
    # --- machine learning ---
    "ml_contamination": 0.02,
    # --- RBA subset / storage limits ---
    "rba_target_rows": 400_000,        # approximate size of the user-level subset
    "rba_min_user_events": 20,         # sampled users need enough history for per-user baselines
    "rba_max_user_events": 20_000,     # skip extreme outliers (bots) when sampling normal users
    "db_alerts_per_incident": 200,     # keep the database small: top alerts per incident
    "watchlist_max": 5000,
    "incident_list_max": 10,           # users / IPs shown per incident
}

# country -> (latitude, longitude, UTC offset hours, typical IPv4 first octets)
COUNTRY_INFO = {
    "IN": (20.6, 78.9, 5.5, [103, 106, 117, 49]),
    "US": (39.8, -98.6, -5.0, [24, 66, 98, 73]),
    "DE": (51.2, 10.4, 1.0, [84, 91, 217, 88]),
    "SG": (1.35, 103.8, 8.0, [116, 175, 202]),
    "GB": (55.4, -3.4, 0.0, [81, 86, 92]),
    "RU": (61.5, 105.3, 3.0, [5, 46, 95, 178]),
    "CN": (35.9, 104.2, 8.0, [36, 58, 111, 223]),
    "BR": (-14.2, -51.9, -3.0, [177, 186, 200]),
    "NG": (9.1, 8.7, 1.0, [41, 105, 197]),
}


# Approximate country centres (lat, lon) for impossible-travel distances on real datasets.
# Countries not listed here are skipped by the travel rule (speed = 0).
COUNTRY_CENTROIDS = {
    "AF": (33.9, 67.7), "AL": (41.2, 20.2), "DZ": (28.0, 1.7), "AO": (-11.2, 17.9), "AR": (-38.4, -63.6),
    "AM": (40.1, 45.0), "AU": (-25.3, 133.8), "AT": (47.5, 14.6), "AZ": (40.1, 47.6), "BD": (23.7, 90.4),
    "BY": (53.7, 27.9), "BE": (50.5, 4.5), "BO": (-16.3, -63.6), "BA": (43.9, 17.7), "BR": (-14.2, -51.9),
    "BG": (42.7, 25.5), "KH": (12.6, 104.9), "CM": (7.4, 12.4), "CA": (56.1, -106.3), "CL": (-35.7, -71.5),
    "CN": (35.9, 104.2), "CO": (4.6, -74.3), "CR": (9.7, -83.8), "HR": (45.1, 15.2), "CU": (21.5, -77.8),
    "CY": (35.1, 33.4), "CZ": (49.8, 15.5), "DK": (56.3, 9.5), "DO": (18.7, -70.2), "EC": (-1.8, -78.2),
    "EG": (26.8, 30.8), "SV": (13.8, -88.9), "EE": (58.6, 25.0), "ET": (9.1, 40.5), "FI": (61.9, 25.7),
    "FR": (46.2, 2.2), "GE": (42.3, 43.4), "DE": (51.2, 10.4), "GH": (7.9, -1.0), "GR": (39.1, 21.8),
    "GT": (15.8, -90.2), "HN": (15.2, -86.2), "HK": (22.3, 114.2), "HU": (47.2, 19.5), "IS": (65.0, -19.0),
    "IN": (20.6, 78.9), "ID": (-0.8, 113.9), "IR": (32.4, 53.7), "IQ": (33.2, 43.7), "IE": (53.4, -8.2),
    "IL": (31.0, 34.9), "IT": (41.9, 12.6), "JM": (18.1, -77.3), "JP": (36.2, 138.3), "JO": (30.6, 36.2),
    "KZ": (48.0, 66.9), "KE": (0.0, 37.9), "KR": (35.9, 127.8), "KW": (29.3, 47.5), "KG": (41.2, 74.8),
    "LV": (56.9, 24.6), "LB": (33.9, 35.9), "LT": (55.2, 23.9), "LU": (49.8, 6.1), "MY": (4.2, 102.0),
    "MX": (23.6, -102.6), "MD": (47.4, 28.4), "MN": (46.9, 103.8), "MA": (31.8, -7.1), "MM": (21.9, 96.0),
    "NP": (28.4, 84.1), "NL": (52.1, 5.3), "NZ": (-40.9, 174.9), "NI": (12.9, -85.2), "NG": (9.1, 8.7),
    "NO": (60.5, 8.5), "OM": (21.5, 55.9), "PK": (30.4, 69.3), "PA": (8.5, -80.8), "PY": (-23.4, -58.4),
    "PE": (-9.2, -75.0), "PH": (12.9, 121.8), "PL": (51.9, 19.1), "PT": (39.4, -8.2), "PR": (18.2, -66.6),
    "QA": (25.4, 51.2), "RO": (45.9, 25.0), "RU": (61.5, 105.3), "SA": (23.9, 45.1), "RS": (44.0, 21.0),
    "SG": (1.35, 103.8), "SK": (48.7, 19.7), "SI": (46.2, 15.0), "ZA": (-30.6, 22.9), "ES": (40.5, -3.7),
    "LK": (7.9, 80.8), "SE": (60.1, 18.6), "CH": (46.8, 8.2), "SY": (34.8, 39.0), "TW": (23.7, 121.0),
    "TH": (15.9, 101.0), "TN": (33.9, 9.5), "TR": (39.0, 35.2), "UA": (48.4, 31.2), "AE": (23.4, 53.8),
    "GB": (55.4, -3.4), "US": (39.8, -98.6), "UY": (-32.5, -55.8), "UZ": (41.4, 64.6), "VE": (6.4, -66.6),
    "VN": (14.1, 108.3), "YE": (15.6, 48.5), "ZM": (-13.1, 27.8), "ZW": (-19.0, 29.2), "TZ": (-6.4, 34.9),
    "UG": (1.4, 32.3), "SD": (12.9, 30.2), "SN": (14.5, -14.5), "CI": (7.5, -5.5), "LY": (26.3, 17.2),
    "BH": (26.0, 50.6), "MT": (35.9, 14.4), "MK": (41.6, 21.7), "ME": (42.7, 19.4), "MU": (-20.3, 57.6),
    "NA": (-22.6, 17.1), "BW": (-22.3, 24.7), "MZ": (-18.7, 35.5), "MG": (-18.8, 46.9), "KP": (40.3, 127.5),
}
COUNTRY_LATLON = {**COUNTRY_CENTROIDS, **{k: (v[0], v[1]) for k, v in COUNTRY_INFO.items()}}


# ==== SECTION: generate ====
def _rand_ip(rng, country):
    prefix = int(rng.choice(COUNTRY_INFO[country][3]))
    return f"{prefix}.{rng.integers(0, 256)}.{rng.integers(0, 256)}.{rng.integers(1, 255)}"


def generate_logs(n_users=25, days=30, seed=42, start="2026-09-01", messy=True):
    """Simulated cloud authentication service.

    Produces normal behaviour, realistic 'hard negatives' (travellers, a VPN user,
    forgotten passwords, new devices, shared office IP) and labelled attacks.
    Columns starting with gt_ are GROUND TRUTH used only for evaluation; the detector
    never reads them.
    """
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start, tz="UTC")
    rows = []
    aid = {"n": 0}

    def add(ts, user, ip, country, device, success, atype="normal", attack_id=0):
        rows.append((ts, user, ip, country, device, int(success),
                     0 if atype == "normal" else 1, atype, attack_id))

    def new_attack():
        aid["n"] += 1
        return aid["n"]

    # ---- user directory ----
    home = rng.choice(["IN", "US", "DE", "SG", "GB"], size=n_users, p=[.5, .2, .12, .12, .06])
    office = {c: _rand_ip(rng, c) for c in set(home)}          # shared office NAT address
    users = []
    for i in range(n_users):
        c = str(home[i])
        users.append({
            "user": f"user{i + 1:02d}", "country": c,
            "devices": ["laptop", "phone"] if rng.random() < .8 else ["laptop"],
            "ips": [office[c] if rng.random() < .5 else _rand_ip(rng, c), _rand_ip(rng, c)],
            "mean_hour": float(rng.normal(12, 1.5)),
            "trip": None, "vpn": False, "swap_day": None,
        })
    for i in (3, 7):                                            # two business travellers
        tc = str(rng.choice([c for c in ["US", "DE", "SG", "GB"] if c != users[i]["country"]]))
        users[i].update(trip=(10, 13), trip_country=tc, trip_ip=_rand_ip(rng, tc))
    users[11]["vpn"] = True                                     # one VPN user
    for i in rng.choice(n_users, size=6, replace=False):        # six users replace their laptop
        users[int(i)]["swap_day"] = int(rng.integers(5, 25))

    # ---- normal behaviour ----
    for u in users:
        off_home = COUNTRY_INFO[u["country"]][2]
        for d in range(days):
            if u["trip"] and d in (u["trip"][0] - 1, u["trip"][1] + 1):
                continue                                         # travel days: no logins
            weekday = (t0 + pd.Timedelta(days=d)).weekday() < 5
            for _ in range(rng.poisson(6 if weekday else 1.5)):
                country, off = u["country"], off_home
                if u["trip"] and u["trip"][0] <= d <= u["trip"][1]:
                    country, ip = u["trip_country"], u["trip_ip"]
                    off = COUNTRY_INFO[country][2]
                elif u["vpn"] and rng.random() < .02:
                    country = str(rng.choice([c for c in ["US", "DE", "SG", "GB"] if c != u["country"]]))
                    ip = _rand_ip(rng, country)
                elif rng.random() < .8:
                    ip = u["ips"][int(rng.integers(len(u["ips"])))]
                else:
                    ip = _rand_ip(rng, country)                  # dynamic ISP address
                device = u["devices"][0] if (len(u["devices"]) == 1 or rng.random() < .7) else u["devices"][1]
                if u["swap_day"] is not None and d >= u["swap_day"] and device == "laptop":
                    device = "laptop2"
                local = float(np.clip(rng.normal(u["mean_hour"], 2.0), 6, 20))
                ts = t0 + pd.Timedelta(days=d, hours=local - off)
                r = rng.random()
                n_fail = 1 if r < .025 else (int(rng.integers(2, 5)) if r < .03 else 0)
                for _f in range(n_fail):                         # ordinary typos
                    add(ts, u["user"], ip, country, device, 0)
                    ts += pd.Timedelta(seconds=float(rng.uniform(10, 40)))
                add(ts, u["user"], ip, country, device, 1)

    # a few legitimate "forgot my password" bursts (hard negatives for the brute-force rule)
    for _ in range(3):
        u = users[int(rng.integers(n_users))]
        ts = t0 + pd.Timedelta(days=float(rng.uniform(6, 29)))
        for _f in range(int(rng.integers(5, 7))):
            add(ts, u["user"], u["ips"][0], u["country"], u["devices"][0], 0)
            ts += pd.Timedelta(seconds=float(rng.uniform(15, 45)))
        add(ts, u["user"], u["ips"][0], u["country"], u["devices"][0], 1)

    # ---- attacks (labelled) ----
    def when():
        return t0 + pd.Timedelta(days=float(rng.uniform(4, 29)))

    def attacker():
        c = str(rng.choice(["RU", "CN", "BR", "NG"]))
        return c, _rand_ip(rng, c)

    def a_device():
        return str(rng.choice(["unknown", "unknown", "tablet", "laptop", "phone"]))

    for _ in range(6):                                           # brute force
        a, (c, ip) = new_attack(), attacker()
        u, ts = users[int(rng.integers(n_users))], when()
        for _f in range(int(rng.integers(10, 40))):
            add(ts, u["user"], ip, c, "unknown", 0, "brute_force", a)
            ts += pd.Timedelta(seconds=float(rng.uniform(2, 15)))

    for _ in range(4):                                           # password spraying
        a, (c, ip) = new_attack(), attacker()
        ts = when()
        for idx in rng.choice(n_users, size=int(rng.integers(6, 13)), replace=False):
            for _f in range(int(rng.integers(1, 3))):
                add(ts, users[int(idx)]["user"], ip, c, "unknown", 0, "password_spray", a)
                ts += pd.Timedelta(seconds=float(rng.uniform(10, 40)))

    for _ in range(5):                                           # failures then success (compromise)
        a, (c, ip) = new_attack(), attacker()
        u, ts, dev = users[int(rng.integers(n_users))], when(), a_device()
        for _f in range(int(rng.integers(3, 9))):
            add(ts, u["user"], ip, c, dev, 0, "compromise", a)
            ts += pd.Timedelta(seconds=float(rng.uniform(5, 30)))
        add(ts, u["user"], ip, c, dev, 1, "compromise", a)

    for _ in range(5):                                           # impossible travel (stolen credentials)
        a = new_attack()
        u, ts = users[int(rng.integers(n_users))], when()
        add(ts, u["user"], u["ips"][0], u["country"], u["devices"][0], 1)          # genuine user
        c = str(rng.choice([x for x in ["RU", "CN", "BR", "NG", "US"] if x != u["country"]]))
        add(ts + pd.Timedelta(minutes=float(rng.uniform(5, 45))), u["user"], _rand_ip(rng, c), c,
            a_device(), 1, "impossible_travel", a)

    for _ in range(5):                                           # odd-hour login from a new place
        a = new_attack()
        u = users[int(rng.integers(n_users))]
        off = COUNTRY_INFO[u["country"]][2]
        ts = t0 + pd.Timedelta(days=int(rng.integers(5, 29)), hours=float(rng.uniform(0.5, 4.0)) - off)
        add(ts, u["user"], _rand_ip(rng, u["country"]), u["country"], a_device(), 1, "odd_hour", a)

    for _ in range(5):                                           # subtle: new device+IP, normal-looking hour
        a = new_attack()
        u = users[int(rng.integers(n_users))]
        off = COUNTRY_INFO[u["country"]][2]
        local = u["mean_hour"] + float(rng.choice([-1, 1])) * float(rng.uniform(3.5, 5.5))
        ts = t0 + pd.Timedelta(days=int(rng.integers(5, 29)), hours=local - off)
        add(ts, u["user"], _rand_ip(rng, u["country"]), u["country"], "tablet", 1, "subtle_takeover", a)

    for _ in range(3):                                           # slow, low-volume spraying
        a, (c, ip) = new_attack(), attacker()
        ts = when()
        for idx in rng.choice(n_users, size=int(rng.integers(15, 22)), replace=False):
            add(ts, users[int(idx)]["user"], ip, c, "unknown", 0, "slow_spray", a)
            ts += pd.Timedelta(minutes=float(rng.uniform(20, 50)))

    df = pd.DataFrame(rows, columns=["timestamp", "user", "ip", "country", "device", "success",
                                     "gt_label", "gt_attack_type", "gt_attack_id"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "event_id", np.arange(1, len(df) + 1))

    # ---- real-world mess (so validation code has something to do) ----
    if messy:
        out = df.copy()
        out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        normal = out.index[out["gt_label"] == 0].to_numpy()
        for i in rng.choice(len(out), size=int(.15 * len(out)), replace=False):   # local-time offsets
            ts = df.at[i, "timestamp"].tz_convert("Asia/Kolkata")
            out.at[i, "timestamp"] = ts.strftime("%Y-%m-%dT%H:%M:%S%z")
        out = out.astype({"user": object, "ip": object})
        out.loc[rng.choice(normal, 15, replace=False), "user"] = np.nan
        out.loc[rng.choice(normal, 10, replace=False), "ip"] = np.nan
        out.loc[rng.choice(normal, 10, replace=False), "ip"] = "999.12.1.7"
        out.loc[rng.choice(normal, 5, replace=False), "timestamp"] = "not-a-time"
        dup = out.sample(frac=.01, random_state=seed)
        out = pd.concat([out, dup], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
        return out
    return df


# ==== SECTION: preprocess ====
def _is_valid_ip(value):
    try:
        ipaddress.ip_address(str(value).strip())
        return True
    except ValueError:
        return False


def preprocess(raw):
    """Validate, normalise (UTC) and de-duplicate raw login records. Returns (clean_df, report)."""
    report = {"raw_rows": len(raw)}
    df = raw.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce", format="ISO8601")
    df["success"] = pd.to_numeric(df["success"], errors="coerce")
    for col in ("user", "ip"):
        df[col] = df[col].astype("string").str.strip()

    n = len(df)
    df = df.dropna(subset=["timestamp", "user", "ip", "success"])
    df = df[df["user"] != ""]
    report["dropped_missing_or_bad_fields"] = n - len(df)

    n = len(df)
    ok_ip = {v: _is_valid_ip(v) for v in df["ip"].unique()}
    df = df[df["ip"].map(ok_ip)]
    report["dropped_invalid_ip"] = n - len(df)

    df["success"] = df["success"].astype(int).clip(0, 1)
    df["device"] = df["device"].fillna("unknown") if "device" in df else "unknown"
    df["country"] = df["country"].fillna("UNK") if "country" in df else "UNK"
    df["asn"] = df["asn"].fillna("UNK").astype(str) if "asn" in df else "UNK"

    n = len(df)
    df = df.drop_duplicates(subset=["timestamp", "user", "ip", "device", "success"])
    report["dropped_duplicates"] = n - len(df)

    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if "event_id" not in df:                       # real logs often have no id column
        df.insert(0, "event_id", np.arange(1, len(df) + 1))
    report["clean_rows"] = len(df)
    return df, report


# ==== SECTION: features ====
def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def _window_counts(t, flag, window_s):
    """For each event: number of flagged events in the window (t - W, t], including itself."""
    cum = np.concatenate([[0], np.cumsum(flag)])
    left = np.searchsorted(t, t - window_s, side="right")
    return cum[np.arange(len(t)) + 1] - cum[left]


def _distinct_in_window(t, flag, codes, window_s):
    """For each FAILED event: number of distinct accounts with a failed event in (t - W, t].
    Two-pointer sliding window, O(n) (the old per-event set() was too slow for real datasets)."""
    t, flag, codes = t.tolist(), flag.tolist(), codes.tolist()
    out = np.zeros(len(t), int)
    counts, left, distinct = {}, 0, 0
    for k in range(len(t)):
        if flag[k]:
            c = codes[k]
            counts[c] = counts.get(c, 0) + 1
            if counts[c] == 1:
                distinct += 1
        while t[left] <= t[k] - window_s:
            if flag[left]:
                c = codes[left]
                counts[c] -= 1
                if counts[c] == 0:
                    distinct -= 1
            left += 1
        if flag[k]:
            out[k] = distinct
    return out


def build_features(df, cfg=CONFIG):
    """Turn clean login events into detection features.

    Every feature only looks at the CURRENT and PAST events, so the same code works
    in a streaming setting (no look-ahead / data leakage).
    """
    f = df.copy()
    if "asn" not in f:
        f["asn"] = "UNK"
    n = len(f)
    t = (f["timestamp"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
    failed = (f["success"] == 0).to_numpy().astype(int)
    w_short, w_long = cfg["window_minutes"] * 60.0, cfg["long_window_hours"] * 3600.0

    # -- burst / volume features --------------------------------------------------
    fails_user = np.zeros(n, int)
    for _, idx in f.groupby("user").indices.items():
        fails_user[idx] = _window_counts(t[idx], failed[idx], w_short)

    codes = pd.factorize(f["user"])[0]
    fails_ip10, fails_ip24 = np.zeros(n, int), np.zeros(n, int)
    dist10, dist24 = np.zeros(n, int), np.zeros(n, int)
    for _, idx in f.groupby("ip").indices.items():
        if len(idx) == 1:                                      # fast path: address seen only once
            fails_ip10[idx] = fails_ip24[idx] = dist10[idx] = dist24[idx] = failed[idx]
            continue
        ti, fi, ci = t[idx], failed[idx], codes[idx]
        fails_ip10[idx] = _window_counts(ti, fi, w_short)
        fails_ip24[idx] = _window_counts(ti, fi, w_long)
        if fi.sum() == 0:
            continue
        dist10[idx] = _distinct_in_window(ti, fi, ci, w_short)
        dist24[idx] = _distinct_in_window(ti, fi, ci, w_long)
    f["fails_user_10m"], f["fails_ip_10m"], f["distinct_users_ip_10m"] = fails_user, fails_ip10, dist10
    f["fails_ip_24h"], f["distinct_users_ip_24h"] = fails_ip24, dist24
    f["log_ip_fails"] = np.log1p(fails_ip24)

    # -- per-user history ("is this normal FOR THIS USER?") -----------------------
    s = (f["success"] == 1).astype(int)
    f["prior_successes"] = s.groupby(f["user"]).cumsum() - s
    known = f["prior_successes"] >= cfg["min_history"]
    succ = f["success"] == 1
    for col, name in (("ip", "new_ip"), ("device", "new_device"), ("country", "new_country"), ("asn", "new_asn")):
        first = ~f.loc[succ].duplicated(subset=["user", col])
        f[name] = 0
        f.loc[first[first].index, name] = 1
        f.loc[~known, name] = 0

    first_c = (~f.loc[succ].duplicated(subset=["user", "country"])).astype(int)
    seen = first_c.groupby(f.loc[succ, "user"]).cumsum()
    f["countries_seen_before"] = 0
    f.loc[succ, "countries_seen_before"] = seen - first_c

    hr = f["timestamp"].dt.hour + f["timestamp"].dt.minute / 60
    ang = 2 * np.pi * hr / 24
    cs = (np.sin(ang) * s).groupby(f["user"]).cumsum() - np.sin(ang) * s
    cc = (np.cos(ang) * s).groupby(f["user"]).cumsum() - np.cos(ang) * s
    mean_hr = (np.arctan2(cs, cc) % (2 * np.pi)) * 24 / (2 * np.pi)
    diff = (hr - mean_hr).abs()
    f["hour_dev"] = np.minimum(diff, 24 - diff).where(known, 0.0)

    # -- impossible travel: speed needed between consecutive successful logins ----
    sd = f.loc[succ, ["user", "timestamp", "country"]].copy()
    g = sd.groupby("user")
    sd["prev_country"], sd["prev_time"] = g["country"].shift(1), g["timestamp"].shift(1)
    hours = (sd["timestamp"] - sd["prev_time"]).dt.total_seconds() / 3600
    lat_map = {k: v[0] for k, v in COUNTRY_LATLON.items()}
    lon_map = {k: v[1] for k, v in COUNTRY_LATLON.items()}
    lat = lambda c: c.map(lat_map).astype(float)
    lon = lambda c: c.map(lon_map).astype(float)
    dist = haversine_km(lat(sd["prev_country"]), lon(sd["prev_country"]), lat(sd["country"]), lon(sd["country"]))
    f["travel_speed_kmh"] = 0.0
    f.loc[sd.index, "travel_speed_kmh"] = (dist / np.maximum(hours, 1 / 60)).fillna(0.0)
    f["log_speed"] = np.log1p(f["travel_speed_kmh"].clip(upper=20000))
    f["failed"] = failed

    # -- familiarity & reputation features (PAST events only, no labels) -----------
    # share_*: how often this user used this IP / device / network / country before (0 = never, 1 = always)
    u_prior = f.groupby("user").cumcount()
    for col in ("ip", "device", "asn", "country"):
        f[f"share_{col}"] = (f.groupby(["user", col]).cumcount() / np.maximum(u_prior, 1)).where(u_prior > 0, 0.0)
    # *_fail_rate: smoothed share of failed logins seen so far from this network / country / IP / user
    fs = pd.Series(failed, index=f.index)
    for col, name in (("asn", "asn"), ("country", "country"), ("ip", "ip"), ("user", "user")):
        n_prior = f.groupby(col).cumcount()
        f_prior = fs.groupby(f[col]).cumsum() - fs
        f[f"{name}_fail_rate"] = (f_prior + 1) / (n_prior + 2)
        if name in ("ip", "asn"):
            f[f"log_{name}_prior"] = np.log1p(n_prior)
    # round-trip time (RBA only): unusual latency often means a proxy / different network path
    if "rtt" in f:
        rtt = pd.to_numeric(f["rtt"], errors="coerce")
        has = rtt.notna().astype(int)
        s_prior = rtt.fillna(0).groupby(f["user"]).cumsum() - rtt.fillna(0)
        n_rtt = has.groupby(f["user"]).cumsum() - has
        f["rtt_log"] = np.log1p(rtt)
        f["rtt_dev"] = np.log1p(rtt) - np.log1p(s_prior / n_rtt.where(n_rtt > 0))
    else:
        f["rtt_log"] = f["rtt_dev"] = np.nan
    for col, name in (("device", "device_code"), ("country", "country_code")):   # frequency-ranked category codes
        rank = {v: i for i, v in enumerate(f[col].value_counts().index[:60])}
        f[name] = f[col].map(rank).fillna(60).astype(float)
    return f


# ==== SECTION: ml ====
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


# ==== SECTION: supervised ====
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


# ==== SECTION: scoring ====
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


# ==== SECTION: incidents ====
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


# ==== SECTION: storage ====
HASH_FIELDS = ["incident_id", "start", "end", "n_alerts", "max_score", "risk_level", "users", "ips", "rules", "summary"]
STATUSES = ("new", "confirmed", "false_positive", "investigating")


def record_hash(row):
    """SHA-256 fingerprint of an incident's immutable fields (tamper evidence)."""
    payload = json.dumps({k: str(row[k]) for k in HASH_FIELDS}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def mask_ip(ip, keep_octets=2):
    """Privacy: show only the network part of an address in dashboards and reports."""
    parts = str(ip).split(".")
    return ".".join(parts[:keep_octets] + ["*"] * (len(parts) - keep_octets)) if len(parts) == 4 else "***"


def save_to_db(db_path, alerts, incidents, meta=None, watchlist=None, extra_tables=None):
    """Persist alerts + incidents in SQLite (parameter-free pandas writer; no user input is ever
    concatenated into SQL). Passwords/tokens are never part of the schema."""
    cols = ["event_id", "timestamp", "user", "ip", "country", "device", "success",
            "risk_score", "risk_level", "rules", "reasons", "incident_id"]
    al = alerts[cols].copy()
    al["timestamp"] = al["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    inc = incidents.copy()
    for c in ("start", "end"):
        inc[c] = inc[c].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    inc["record_hash"] = inc.apply(record_hash, axis=1)
    inc["status"], inc["reviewed_by"] = "new", ""
    con = sqlite3.connect(db_path)
    try:
        al.to_sql("alerts", con, if_exists="replace", index=False)
        inc.to_sql("incidents", con, if_exists="replace", index=False)
        if watchlist is not None:
            wl = watchlist[cols[:-1]].copy()
            wl["timestamp"] = wl["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            wl.to_sql("watchlist", con, if_exists="replace", index=False)
        pd.DataFrame([meta or {}]).to_sql("run_meta", con, if_exists="replace", index=False)
        for name, tbl in (extra_tables or {}).items():          # evaluation tables for the dashboard
            tbl.to_sql(name, con, if_exists="replace", index=False)
        con.execute("CREATE TABLE IF NOT EXISTS audit_log (ts TEXT, reviewer TEXT, incident_id INTEGER, old_status TEXT, new_status TEXT)")
        con.commit()
    finally:
        con.close()


def update_status(db_path, incident_id, new_status, reviewer):
    """Administrator review action. Parameterised SQL + allow-listed values + audit trail."""
    if new_status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    con = sqlite3.connect(db_path)
    try:
        old = con.execute("SELECT status FROM incidents WHERE incident_id = ?", (int(incident_id),)).fetchone()
        if old is None:
            raise KeyError(f"no such incident: {incident_id}")
        con.execute("UPDATE incidents SET status = ?, reviewed_by = ? WHERE incident_id = ?",
                    (new_status, reviewer, int(incident_id)))
        con.execute("INSERT INTO audit_log VALUES (?,?,?,?,?)",
                    (datetime.now(timezone.utc).isoformat(), reviewer, int(incident_id), old[0], new_status))
        con.commit()
    finally:
        con.close()


def verify_integrity(db_path):
    """Return incident ids whose stored fingerprint no longer matches their content."""
    con = sqlite3.connect(db_path)
    try:
        inc = pd.read_sql_query("SELECT * FROM incidents", con)
    finally:
        con.close()
    return [int(r.incident_id) for r in inc.itertuples() if record_hash(r._asdict()) != r.record_hash]


# ==== SECTION: evaluation ====
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


# ==== SECTION: RBA loader ====
RBA_USECOLS = ["Login Timestamp", "User ID", "Round-Trip Time [ms]", "IP Address", "Country", "ASN", "Browser Name and Version",
               "OS Name and Version", "Device Type", "Login Successful", "Is Attack IP", "Is Account Takeover"]
_CSV_KW = dict(true_values=["True", "true"], false_values=["False", "false"],
               keep_default_na=False, na_values=[""])      # 'NA' (Namibia) must not become NaN


def _flag(s):
    """True/False column -> bool, tolerant of strings."""
    return s.astype(str).str.strip().str.lower().eq("true")


def _strip_version(s):
    """'Chrome 89.0.4389' -> 'Chrome', 'Windows 10' -> 'Windows'. Browser auto-updates must not look like a new device."""
    return s.fillna("unknown").astype(str).str.replace(r"[\s\d\.]+$", "", regex=True).str.strip().replace("", "unknown")


def load_rba_subset(csv_path, target_rows=None, min_events=None, max_events=None,
                    subset_path="data/rba_subset.csv", rebuild=False, cfg=CONFIG, chunksize=1_000_000):
    """Build a USER-LEVEL subset of the 31M-row RBA file (reads it in chunks, low memory).

    Why by user and not by random row: every feature compares a login with that user's own
    history, so a sampled user must keep ALL of their logins.

    Subset = every user with a confirmed account takeover (never lose an attack)
           + a deterministic pseudo-random sample of other users with enough history,
             until about `target_rows` rows are reached.
    The device fingerprint is browser + OS + device type (versions stripped); the network is the ASN.
    The result is cached in `subset_path`, so later runs skip the two slow passes. A cache written by
    an older version of this code (no ASN column) is rebuilt automatically.
    Returns (raw_dataframe, info_dict).
    """
    target_rows = target_rows or cfg["rba_target_rows"]
    min_events = min_events or cfg["rba_min_user_events"]
    max_events = max_events or cfg["rba_max_user_events"]
    import os
    if os.path.exists(subset_path) and not rebuild:
        raw = pd.read_csv(subset_path, dtype={"user": str, "asn": str}, keep_default_na=False, na_values=[""])
        if {"asn", "rtt"} <= set(raw.columns):
            print(f"[loader] reusing cached subset {subset_path} ({len(raw):,} rows). Use --rebuild to redo it.")
            return raw, {"source": csv_path, "subset_rows": len(raw), "cached": True}
        print("[loader] cached subset is from an older version (no ASN / browser / OS / RTT): rebuilding it.")

    # ---- pass 1: events per user + which users have a confirmed takeover ----
    print("[loader] pass 1/2: counting events per user (reads the whole file once)...")
    parts, ato_users, seen = [], set(), 0
    for ch in pd.read_csv(csv_path, usecols=["User ID", "Is Account Takeover"],
                          dtype={"User ID": "int64"}, chunksize=chunksize, **_CSV_KW):
        parts.append(ch["User ID"].value_counts())
        ato_users.update(ch.loc[_flag(ch["Is Account Takeover"]), "User ID"].unique().tolist())
        seen += len(ch)
        if len(parts) >= 4:
            parts = [pd.concat(parts).groupby(level=0).sum()]
        print(f"[loader]   {seen:,} rows scanned", end="\r", flush=True)
    counts = pd.concat(parts).groupby(level=0).sum().astype("int64")
    print(f"\n[loader] {seen:,} rows, {len(counts):,} users, {len(ato_users)} users with a takeover")

    is_ato = counts.index.isin(list(ato_users))
    ato_rows = int(counts[is_ato].sum())
    pool = counts[~is_ato & (counts >= min_events) & (counts <= max_events)]
    order = np.argsort(np.mod(pool.index.to_numpy(), 1_000_003), kind="stable")   # fixed pseudo-random order
    cum = np.cumsum(pool.to_numpy()[order])
    take = order[cum <= max(target_rows - ato_rows, 0)]
    chosen = np.concatenate([np.array(sorted(ato_users), dtype="int64"), pool.index.to_numpy()[take]])
    print(f"[loader] chosen: {len(ato_users)} takeover users ({ato_rows:,} rows) + {len(take):,} sampled users "
          f"with {min_events}-{max_events} events")

    # ---- pass 2: pull the chosen users' rows ----
    print("[loader] pass 2/2: extracting the chosen users...")
    keep = []
    for ch in pd.read_csv(csv_path, usecols=RBA_USECOLS, dtype={"User ID": "int64", "ASN": "str"},
                          chunksize=chunksize, **_CSV_KW):
        ch = ch[ch["User ID"].isin(chosen)]
        if len(ch):
            fingerprint = (_strip_version(ch["Browser Name and Version"]) + " | "
                           + _strip_version(ch["OS Name and Version"]) + " | "
                           + ch["Device Type"].fillna("unknown").astype(str))
            keep.append(pd.DataFrame({
                "timestamp": ch["Login Timestamp"], "user": ch["User ID"].astype(str), "ip": ch["IP Address"],
                "country": ch["Country"], "asn": ch["ASN"].fillna("UNK"), "device": fingerprint,
                "rtt": pd.to_numeric(ch["Round-Trip Time [ms]"], errors="coerce"),
                "success": _flag(ch["Login Successful"]).astype(int),
                "gt_attack_ip": _flag(ch["Is Attack IP"]).astype(int),
                "gt_label": _flag(ch["Is Account Takeover"]).astype(int)}))
    raw = pd.concat(keep, ignore_index=True)
    os.makedirs(os.path.dirname(subset_path) or ".", exist_ok=True)
    raw.to_csv(subset_path, index=False)
    info = {"source": csv_path, "subset_rows": len(raw), "subset_users": int(raw["user"].nunique()),
            "takeover_users": len(ato_users), "takeover_rows": int(raw["gt_label"].sum()),
            "attack_ip_rows": int(raw["gt_attack_ip"].sum()), "distinct_devices": int(raw["device"].nunique()),
            "distinct_asn": int(raw["asn"].nunique()), "cached": False}
    print(f"[loader] subset saved to {subset_path}: {info}")
    return raw, info


# ==== SECTION: RBA evaluation ====
def _split_by_user(users):
    """Deterministic 50/50 split by user: 'train' (fit the ML model, run threshold sweeps) vs
    'test' (final numbers). A user's logins never straddle both halves."""
    h = pd.util.hash_pandas_object(pd.Series(users).astype(str), index=False).to_numpy() % 2
    return np.where(h == 0, "train", "test")


def evaluate_rba_events(scored, cfg=CONFIG, mask=None):
    """Event-level precision/recall/FPR for each strategy, against BOTH real labels.
    'Account takeover' = confirmed by the service's incident-response team (strict, very rare).
    'Attack IP' = address found in a known-attacker list (broad)."""
    m = np.ones(len(scored), bool) if mask is None else np.asarray(mask, bool)
    rules = score_events(scored, cfg, use_ml=False, use_sup=False, explain=False)
    hyb_ml = score_events(scored, cfg, use_sup=False, explain=False)
    full = score_events(scored, cfg, explain=False)
    zero = np.zeros(len(scored), bool)
    strategies = {
        "Rules only (alerts: Medium+)": rules["alert"].to_numpy(),
        "ML only (Isolation Forest)": scored["ml_flag"].to_numpy(),
        "Hybrid rules+ML (alerts: Medium+)": hyb_ml["alert"].to_numpy(),
        "Supervised: attack-IP model": scored["sup_ip_flag"].to_numpy() if "sup_ip_flag" in scored else zero,
        "Supervised: takeover model": scored["sup_ato_flag"].to_numpy() if "sup_ato_flag" in scored else zero,
        "Full hybrid: rules+ML+supervised": full["alert"].to_numpy(),
        "Full hybrid incl. Low watchlist": (full["risk_score"] >= 1).to_numpy(),
    }
    rows = []
    for label, col in (("Account takeover", "gt_label"), ("Attack IP", "gt_attack_ip")):
        y = (scored[col] == 1).to_numpy()[m]
        for name, pred in strategies.items():
            rows.append({"label": label, "strategy": name, "alerts": int(pred[m].sum()),
                         **confusion_metrics(y, pred[m])})
    return pd.DataFrame(rows)


def evaluate_rba_rules(scored, mask=None):
    """Which rule contributes what? (fired events, hit rate on attacker IPs, takeover hits)."""
    m = np.ones(len(scored), bool) if mask is None else np.asarray(mask, bool)
    atk, ato = (scored["gt_attack_ip"] == 1).to_numpy() & m, (scored["gt_label"] == 1).to_numpy() & m
    rows = []
    for c in RULE_COLS + MODEL_COLS:
        fired = scored[c].to_numpy(bool) & m
        n = int(fired.sum())
        rows.append({"rule": c, "events_fired": n,
                     "precision_vs_attack_ip": (fired & atk).sum() / n if n else np.nan,
                     "recall_vs_attack_ip": (fired & atk).sum() / atk.sum() if atk.sum() else np.nan,
                     "takeover_events_hit": int((fired & ato).sum())})
    return pd.DataFrame(rows)


def evaluate_rba_summary(scored, alerts, mask=None):
    """Attack-level view: how many real takeovers did the administrator get to see?"""
    m = np.ones(len(scored), bool) if mask is None else np.asarray(mask, bool)
    ato = (scored["gt_label"] == 1).to_numpy() & m
    alert, watch = scored["alert"].to_numpy(), (scored["risk_score"] >= 1).to_numpy()
    users = scored["user"].to_numpy()
    n_ato_users = len(set(users[ato]))
    g = alerts.groupby("incident_id")[["gt_attack_ip", "gt_label"]].max()
    out = {
        "takeover_events": int(ato.sum()),
        "takeover_events_alerted": int((ato & alert).sum()),
        "takeover_events_on_watchlist_or_alerted": int((ato & watch).sum()),
        "takeover_users": n_ato_users,
        "takeover_users_with_alert": len(set(users[ato & alert])),
        "takeover_users_with_alert_or_watchlist": len(set(users[ato & watch])),
        "incidents_total (all users)": int(len(g)),
        "incidents_containing_attack_ip": int((g["gt_attack_ip"] == 1).sum()),
        "incidents_containing_takeover": int((g["gt_label"] == 1).sum()),
        "incident_share_with_attack_ip": float((g["gt_attack_ip"] == 1).mean()) if len(g) else 0.0,
        "raw_alerts (all users)": int(len(alerts)),
        "alerts_per_incident": float(len(alerts) / len(g)) if len(g) else 0.0,
    }
    return pd.DataFrame({"metric": list(out), "value": [str(round(v, 4)) if isinstance(v, float) else str(v)
                                                       for v in out.values()]})


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


# ==== SECTION: pipeline ====
def run_pipeline(raw=None, cfg=CONFIG, seed=None, holdout=False):
    """holdout=True: split users 50/50 (column 'split'); the ML model is fitted on 'train' users only."""
    seed = cfg["seed"] if seed is None else seed
    cfg = {**cfg, "seed": seed}
    raw = generate_logs(seed=seed) if raw is None else raw
    clean, report = preprocess(raw)
    feat = build_features(clean, cfg)
    fit_mask = None
    if holdout:
        feat["split"] = _split_by_user(feat["user"])
        fit_mask = (feat["split"] == "train").to_numpy()
    feat, model = add_ml_scores(feat, cfg, fit_mask=fit_mask)
    sup_report = None
    if holdout and {"gt_attack_ip", "gt_label"} <= set(feat.columns):     # real labels available (RBA)
        print("[pipeline] training supervised models (cross-fitted by user) ...")
        feat, sup_report = add_supervised_scores(feat, cfg)
    scored = score_events(feat, cfg)
    incidents, alerts = build_incidents(scored, cfg)
    return {"raw": raw, "clean": clean, "report": report, "features": feat, "scored": scored,
            "incidents": incidents, "alerts": alerts, "model": model, "cfg": cfg, "sup_report": sup_report}


def _prepare_db_tables(res, cfg):
    """Trim what is stored so the database stays small enough to deploy."""
    scored, alerts = res["scored"], res["alerts"]
    watch = scored[(scored["risk_score"] > 0) & ~scored["alert"]]
    watch = watch.sort_values(["risk_score", "ml_anomaly"], ascending=False).head(cfg["watchlist_max"])
    alerts_db = (alerts.sort_values("risk_score", ascending=False)
                 .groupby("incident_id").head(cfg["db_alerts_per_incident"]))
    return alerts_db, watch


def main():
    import argparse
    import os
    ap = argparse.ArgumentParser(description="Suspicious login detection (RBA dataset or simulated demo)")
    ap.add_argument("--rba", default="data/rba-dataset.csv", help="path to the RBA csv")
    ap.add_argument("--demo", action="store_true", help="use the simulated Milestone-3 data instead")
    ap.add_argument("--rows", type=int, default=CONFIG["rba_target_rows"], help="approx. subset size")
    ap.add_argument("--min-events", type=int, default=CONFIG["rba_min_user_events"])
    ap.add_argument("--subset", default="data/rba_subset.csv", help="cache file for the subset")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the subset even if cached")
    ap.add_argument("--db", default="alerts.db")
    args = ap.parse_args()
    cfg = CONFIG
    if os.path.exists(args.db):
        os.remove(args.db)                      # fresh run = fresh audit log

    if args.demo:                               # ---- Milestone-3 behaviour ----
        res = run_pipeline()
        res["raw"].to_csv("simulated_login_logs.csv", index=False)
        meta = {"data_source": "simulated", "generated_utc": datetime.now(timezone.utc).isoformat(),
                "clean_rows": len(res["clean"]), **res["report"]}
        watch = res["scored"][(res["scored"]["risk_score"] > 0) & ~res["scored"]["alert"]]
        save_to_db(args.db, res["alerts"], res["incidents"], meta, watchlist=watch)
        print("Data quality report:", res["report"])
        print(evaluate_events(res["features"], res["cfg"]).round(3).to_string())
        summ, by_type, _ = evaluate_incidents(res["scored"], res["incidents"], res["alerts"])
        print(summ)
        print(by_type.round(2).to_string())
        print(f"Saved simulated_login_logs.csv and {args.db} -> run: streamlit run app.py")
        return

    # ---- RBA run ----
    if not os.path.exists(args.rba) and not (os.path.exists(args.subset) and not args.rebuild):
        raise SystemExit(f"Cannot find {args.rba}. Put the RBA csv there or pass --rba <path>.")
    raw, info = load_rba_subset(args.rba, args.rows, args.min_events, subset_path=args.subset,
                                rebuild=args.rebuild, cfg=cfg)
    print(f"\n[pipeline] analysing {len(raw):,} login events ...")
    res = run_pipeline(raw=raw, holdout=True)
    scored, report = res["scored"], res["report"]
    test, train = (scored["split"] == "test").to_numpy(), (scored["split"] == "train").to_numpy()
    print("\nData quality report:", report)
    print(f"Users: {scored['user'].nunique():,} | train users: {scored.loc[train, 'user'].nunique():,} | "
          f"test users: {scored.loc[test, 'user'].nunique():,}")
    print(f"Distinct device fingerprints: {scored['device'].nunique():,} | distinct ASNs: {scored['asn'].nunique():,}")
    print(f"Real labels in subset: takeover events={int((scored['gt_label'] == 1).sum())}, "
          f"attack-IP events={int((scored['gt_attack_ip'] == 1).sum()):,}")

    ev_events = evaluate_rba_events(scored, res["cfg"], mask=test)
    ev_rules = evaluate_rba_rules(scored, mask=test)
    ev_summary = evaluate_rba_summary(scored, res["alerts"], mask=test)
    ev_sweep = sensitivity_sweep(scored, res["cfg"], mask=train)
    print("\n=== Event-level results on TEST users (ML fitted on TRAIN users only) ===")
    print(ev_events.round(4).to_string(index=False))
    ev_sup = res["sup_report"]
    if ev_sup is not None:
        print("\n=== Supervised models (thresholds chosen on TRAIN users only) ===")
        print(ev_sup.round(4).to_string(index=False))
    print("\n=== Per-rule / per-model contribution (TEST users) ===")
    print(ev_rules.round(4).to_string(index=False))
    print("\n=== Takeover coverage / incidents ===")
    print(ev_summary.to_string(index=False))
    print("\n=== Threshold sensitivity (TRAIN users) ===")
    print(ev_sweep.round(4).to_string(index=False))

    alerts_db, watch = _prepare_db_tables(res, cfg)
    if res["incidents"].empty:
        raise SystemExit("No alerts were raised, so there is nothing to store. Check thresholds in CONFIG.")
    meta = {"data_source": "RBA dataset (Wiefling et al. 2022, CC BY 4.0, doi 10.1145/3546069)",
            "generated_utc": datetime.now(timezone.utc).isoformat(), "subset_rows": len(raw),
            "subset_users": int(raw["user"].nunique()), **report}
    save_to_db(args.db, alerts_db, res["incidents"], meta, watchlist=watch,
               extra_tables={"eval_events": ev_events, "eval_rules": ev_rules,
                             "eval_summary": ev_summary, "eval_sensitivity": ev_sweep,
                             **({"eval_supervised": ev_sup} if ev_sup is not None else {})})
    print(f"\nSaved {args.db} ({len(res['incidents']):,} incidents, {len(alerts_db):,} stored alerts, "
          f"{len(watch):,} watchlist rows) -> run: streamlit run app.py")


if __name__ == "__main__":
    main()