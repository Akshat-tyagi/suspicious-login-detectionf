"""End-to-end pipeline and the command-line entry point.  (split from detector.py)"""

from datetime import datetime, timezone
import pandas as pd
from sld.config import CONFIG
from sld.simulate import generate_logs
from sld.preprocess import preprocess
from sld.features import build_features
from sld.models import add_ml_scores, add_supervised_scores
from sld.scoring import build_incidents, score_events
from sld.storage import save_to_db
from sld.evaluation import _split_by_user, evaluate_events, evaluate_incidents, evaluate_rba_events, evaluate_rba_summary
from sld.rba_loader import load_rba_subset


# ---- pipeline ----
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
    takeover = scored["gt_label"] == 1
    print(f"Takeover events: {int(takeover.sum()):,} | train: {int((takeover & train).sum()):,} | "
          f"test: {int((takeover & test).sum()):,}")

    levels = scored["risk_level"].value_counts()
    summary = {
        "Login attempts analysed": len(scored),
        "Suspicious login alerts": int(scored["alert"].sum()),
        "Incidents created": len(res["incidents"]),
        "High-risk logins": int(levels.get("High", 0)),
        "Medium-risk logins": int(levels.get("Medium", 0)),
        "Low-risk logins (watchlist)": int(levels.get("Low", 0)),
    }
    ev_events = evaluate_rba_events(scored, res["cfg"], mask=test)
    ev_summary = evaluate_rba_summary(scored, res["alerts"], mask=test)
    print("\n=== Account takeover evaluation (TEST users) ===")
    print("Suspicious login alerts are evaluated against confirmed account takeover labels.")
    print("An alert without a takeover label may still indicate other suspicious activity.")
    print(ev_events.drop(columns="label").round(4).to_string(index=False))
    print("\n=== Takeover coverage (TEST users) ===")
    print(ev_summary.to_string(index=False))

    alerts_db, watch = _prepare_db_tables(res, cfg)
    if res["incidents"].empty:
        raise SystemExit("No alerts were raised, so there is nothing to store. Check thresholds in CONFIG.")
    meta = {"data_source": "RBA dataset (Wiefling et al. 2022, CC BY 4.0, doi 10.1145/3546069)",
            "generated_utc": datetime.now(timezone.utc).isoformat(), "subset_rows": len(raw),
            "subset_users": int(raw["user"].nunique()),
            "total_alerts": int(scored["alert"].sum()),
            "high_risk_logins": int(levels.get("High", 0)),
            "medium_risk_logins": int(levels.get("Medium", 0)),
            "low_risk_logins": int(levels.get("Low", 0)), **report}
    save_to_db(args.db, alerts_db, res["incidents"], meta, watchlist=watch,
               extra_tables={"eval_events": ev_events, "eval_summary": ev_summary})
    print("\n=== Suspicious login detection summary (ALL users) ===")
    print(pd.DataFrame({"metric": list(summary), "value": list(summary.values())}).to_string(index=False))
    print("Alerts indicate review priority, not confirmed attacks. Evaluation above uses TEST users only.")
