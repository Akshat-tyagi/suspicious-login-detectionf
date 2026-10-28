"""Validate and clean raw login records.  (split from detector.py)"""

import ipaddress
import numpy as np
import pandas as pd


# ---- preprocess ----
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
