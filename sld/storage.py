"""SQLite storage, review workflow, audit log, tamper check, IP masking.  (split from detector.py)"""

import hashlib
import json
import pandas as pd
import sqlite3
from datetime import datetime, timezone


# ---- storage ----
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
