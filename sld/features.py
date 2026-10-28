"""Feature engineering: windows, familiarity, travel speed.  (split from detector.py)"""

import numpy as np
import pandas as pd
from sld.config import CONFIG, COUNTRY_LATLON


# ---- features ----
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
