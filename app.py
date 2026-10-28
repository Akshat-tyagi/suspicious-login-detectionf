"""Simple suspicious-login dashboard. Run with streamlit run app.py.

Set DASHBOARD_PASSWORD in the environment or Streamlit secrets.
ALERTS_DB optionally selects a database; otherwise use alerts.db beside this file.
"""
from contextlib import closing
import hmac
import ipaddress
import os
from pathlib import Path
import re
import sqlite3
import time

import pandas as pd
import streamlit as st

DB_PATH = str(Path(os.environ.get("ALERTS_DB", Path(__file__).with_name("alerts.db"))).resolve())
TABLES = {"alerts", "incidents", "watchlist", "run_meta", "eval_events"}
MAX_ROWS = 500

st.set_page_config(page_title="Suspicious Login Detection", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1250px; padding-top: 2rem;}
[data-testid="stMetric"] {border: 1px solid #b4bbc440; border-radius: 12px; padding: 16px;}
h1 {letter-spacing: -.035em;}
</style>
""", unsafe_allow_html=True)


def connect_ro(path=DB_PATH):
    return sqlite3.connect(Path(path).as_uri() + "?mode=ro", uri=True, timeout=10)


def require_login():
    secret = os.environ.get("DASHBOARD_PASSWORD")
    if not secret:
        try:
            secret = st.secrets.get("DASHBOARD_PASSWORD")
        except (FileNotFoundError, KeyError):
            secret = None
    if not isinstance(secret, str) or not secret:
        st.title("Suspicious Login Detection")
        st.info("Set DASHBOARD_PASSWORD in your environment or Streamlit secrets to enable administrator access.")
        st.code('export DASHBOARD_PASSWORD="your-password"\nstreamlit run app.py', language="bash")
        st.stop()
    # Bind the authenticated session to the configured password without storing it.
    import hashlib
    signature = hashlib.sha256(secret.encode()).hexdigest()
    if st.session_state.get("auth_signature") == signature:
        return
    st.title("Suspicious Login Detection")
    st.write("Sign in to view suspicious login alerts and takeover evaluation.")
    if st.session_state.get("attempts", 0) >= 5:
        st.error("This session has reached its sign-in attempt limit.")
        st.stop()
    with st.form("login"):
        password = st.text_input("Administrator password", type="password")
        submit = st.form_submit_button("Sign in", type="primary")
    if submit:
        if hmac.compare_digest(password.encode(), secret.encode()):
            st.session_state["auth_signature"] = signature
            st.session_state["attempts"] = 0
            st.rerun()
        st.session_state["attempts"] = st.session_state.get("attempts", 0) + 1
        time.sleep(1)
        st.error("Incorrect password. Please try again.")
    st.stop()


@st.cache_data(show_spinner=False, max_entries=30)
def read_table(path, name, stamp):
    if name not in TABLES:
        raise ValueError("Unknown table")
    with closing(connect_ro(path)) as con:
        if not con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone():
            return pd.DataFrame()
        return pd.read_sql_query(f'SELECT * FROM "{name}"', con)


def load(name):
    stat = Path(DB_PATH).stat()
    return read_table(DB_PATH, name, (stat.st_mtime_ns, stat.st_size))


def mask_text(value):
    if not isinstance(value, str):
        return value
    value = re.sub(r"\b(\d{1,3})\.(\d{1,3})\.\d{1,3}\.\d{1,3}\b", r"\1.\2.*.*", value)
    def mask_v6(match):
        try:
            ipaddress.IPv6Address(match.group())
            return "[IPv6 masked]"
        except ValueError:
            return match.group()
    return re.sub(r"(?<![\w:])[0-9a-fA-F]*:[0-9a-fA-F:.]*:[0-9a-fA-F:.]*(?![\w:])", mask_v6, value)


def show(frame):
    try:
        st.dataframe(frame, width="stretch", hide_index=True)
    except TypeError:  # Streamlit versions before stretch-width support.
        st.dataframe(frame, use_container_width=True, hide_index=True)


def number(meta, key, fallback=0):
    value = meta.get(key, fallback)
    return int(value) if pd.notna(value) else int(fallback)

def detection_summary(alerts, incidents, watch, meta):
    st.subheader("Suspicious login detection summary")
    st.caption("All analysed users · predictions from the combined detector")
    cols = st.columns(3)
    cols[0].metric("Logins analysed", f"{number(meta, 'clean_rows', number(meta, 'subset_rows')):,}")
    cols[1].metric("Suspicious login alerts", f"{number(meta, 'total_alerts', incidents.n_alerts.sum()):,}")
    cols[2].metric("Incidents created", f"{len(incidents):,}")
    counts = alerts.risk_level.value_counts()
    complete = all(key in meta for key in ('high_risk_logins', 'medium_risk_logins', 'low_risk_logins'))
    cols = st.columns(3)
    for col, level, key, fallback in zip(cols, ['High', 'Medium', 'Low'],
            ['high_risk_logins', 'medium_risk_logins', 'low_risk_logins'],
            [counts.get('High', 0), counts.get('Medium', 0), len(watch)]):
        label = f"{level}-risk logins" if complete else f"Stored {level.lower()}-risk entries"
        col.metric(label, f"{number(meta, key, fallback):,}")
    st.info("High- and medium-risk logins raise alerts. Low-risk signals remain on the watchlist. An alert means a login needs investigation; it does not confirm an attack.")
    st.caption("An incident groups related alerts for the same user or IP close in time.")
    if not complete:
        st.caption("This older database has limited stored detail rows. Rerun python detector.py to save complete risk-level totals.")


def suspicious_alerts(alerts):
    st.subheader("Suspicious login alerts")
    a, b, c = st.columns(3)
    levels = a.multiselect("Risk level", ['High', 'Medium', 'Low'], default=['High', 'Medium'])
    user = b.text_input("User ID contains")
    outcome = c.selectbox("Login outcome", ['All', 'Successful', 'Failed'])
    view = alerts[alerts.risk_level.isin(levels)]
    if user.strip():
        view = view[view.user.astype(str).str.contains(user.strip(), regex=False)]
    if outcome != 'All':
        view = view[view.success == (1 if outcome == 'Successful' else 0)]
    columns = ['timestamp', 'user', 'ip', 'country', 'success', 'risk_score', 'risk_level', 'reasons']
    top = view.sort_values(['risk_score', 'timestamp'], ascending=False).head(MAX_ROWS)[columns].copy()
    for col in top.columns:
        top[col] = top[col].map(mask_text)
    top['success'] = top.success.map({1: 'Successful', 0: 'Failed'})
    top = top.rename(columns={'timestamp': 'Time (UTC)', 'user': 'User', 'ip': 'Source IP',
        'country': 'Country', 'success': 'Login outcome', 'risk_score': 'Risk score',
        'risk_level': 'Risk level', 'reasons': 'Why it was flagged'})
    st.caption(f"{len(view):,} matching stored alerts · showing {len(top):,}. IP addresses are masked.")
    if top.empty:
        st.info("No alerts match these filters.")
    show(top)
    safe = top.copy()
    for col in safe.columns:
        safe[col] = safe[col].map(lambda v: "'" + v if isinstance(v, str) and v.startswith(('=', '+', '-', '@', '\t', '\r')) else v)
    st.download_button("Download displayed alerts", safe.to_csv(index=False).encode('utf-8'), 'suspicious_logins.csv', 'text/csv')
    st.caption("The database may retain only the highest-scoring alerts per incident. Low-risk watchlist entries are not included in this alert table.")


def takeover_evaluation(ev):
    st.subheader("Account takeover evaluation")
    st.caption("Test users only · detection thresholds selected using training users")
    st.write("We evaluate suspicious login alerts against the dataset's confirmed account takeover labels. This measures takeover detection, rather than every kind of suspicious activity.")
    if ev.empty or 'label' not in ev:
        st.info("No takeover evaluation is saved. Run python detector.py with the RBA dataset first.")
        return
    ev = ev[ev.label == 'Account takeover'].copy()
    if ev.empty:
        st.info("No account takeover results are available in this database.")
        return
    full = ev[ev.strategy == 'Full hybrid: rules+ML+supervised']
    if not full.empty:
        row = full.iloc[0]
        total = int(row.TP + row.FN)
        cols = st.columns(4)
        cols[0].metric("Actual takeovers", total)
        cols[1].metric("Detected", int(row.TP))
        cols[2].metric("Missed", int(row.FN))
        cols[3].metric("Detection rate", f"{row.TP / total * 100:.2f}%" if total else 'N/A')
    names = {'Rules only (alerts: Medium+)': 'Rules only',
        'Isolation Forest only': 'Isolation Forest only',
        'ML only (Isolation Forest)': 'Isolation Forest only',
        'Supervised only (either model)': 'Supervised only',
        'Full hybrid: rules+ML+supervised': 'All methods combined'}
    comparison = ev[ev.strategy.isin(names)].copy()
    if 'Supervised only (either model)' not in set(comparison.strategy):
        st.caption("Rerun python detector.py to generate the updated four-method comparison.")
    comparison['strategy'] = comparison.strategy.map(names)
    for col in ['precision', 'recall', 'false_positive_rate']:
        comparison[col] = (comparison[col] * 100).round(2)
    comparison = comparison.drop(columns='label').rename(columns={
        'strategy': 'Method', 'alerts': 'Flagged logins', 'precision': 'Precision (%)',
        'recall': 'Recall (%)', 'false_positive_rate': 'False positive rate (%)', 'f1': 'F1'})
    show(comparison.round(4))
    st.caption("TP = takeovers flagged · FP = flags without a takeover label · FN = takeovers missed · TN = non-takeover-labelled logins left unflagged.")
    st.caption("Precision: share of flags matching takeover labels. Recall: share of takeovers detected. F1 balances both. Counts overlap across methods; do not add the rows.")
    st.info("A flag without a takeover label may still indicate another suspicious activity, such as failed password guessing. These metrics do not evaluate all suspicious logins. Results describe this sampled dataset, not guaranteed real-world performance.")


def main():
    require_login()
    st.title("Suspicious Login Detection")
    st.caption("Rules + Isolation Forest + supervised learning")
    a, b, _ = st.columns([1, 1, 5])
    if a.button("Refresh results"):
        read_table.clear()
        st.rerun()
    if b.button("Sign out"):
        st.session_state.clear()
        st.rerun()
    if not Path(DB_PATH).is_file():
        st.info("Generate detection results first, then refresh this page.")
        st.code('python detector.py', language='bash')
        st.stop()
    try:
        alerts, incidents, watch, metadata = (load(name) for name in ['alerts', 'incidents', 'watchlist', 'run_meta'])
        required = {'timestamp', 'user', 'ip', 'country', 'success', 'risk_score', 'risk_level', 'reasons'}
        if not required <= set(alerts.columns) or 'n_alerts' not in incidents:
            st.error("This database is missing detection results. Run python detector.py to generate a fresh database.")
            st.stop()
        meta = metadata.iloc[0].to_dict() if not metadata.empty else {}
        if meta.get('generated_utc'):
            stamp = pd.to_datetime(meta['generated_utc'], utc=True, errors='coerce')
            if pd.notna(stamp):
                st.caption('Latest analysis: ' + stamp.strftime('%d %b %Y, %H:%M UTC'))
        summary, details, evaluation = st.tabs(['Detection summary', 'Suspicious alerts', 'Takeover evaluation'])
        with summary:
            detection_summary(alerts, incidents, watch, meta)
        with details:
            suspicious_alerts(alerts)
        with evaluation:
            takeover_evaluation(load('eval_events'))
    except (sqlite3.Error, pd.errors.DatabaseError, OSError):
        st.error("The database could not be read. Refresh after the detector finishes saving, or check ALERTS_DB.")


if __name__ == '__main__':
    main()
