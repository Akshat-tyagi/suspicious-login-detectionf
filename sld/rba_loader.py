"""Load a user-level subset of the RBA dataset.  (split from detector.py)"""

import numpy as np
import pandas as pd
from sld.config import CONFIG


# ---- RBA loader ----
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
