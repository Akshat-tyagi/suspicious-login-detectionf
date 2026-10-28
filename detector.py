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

# detector.py is now a thin entry point: the code lives in the sld/ package.
# Everything is re-exported here, so `import detector as d`, app.py and the tests keep working.

from sld.config import (  # noqa: F401
    CONFIG,
    COUNTRY_CENTROIDS,
    COUNTRY_INFO,
    COUNTRY_LATLON,
)
from sld.simulate import (  # noqa: F401
    _rand_ip,
    generate_logs,
)
from sld.preprocess import (  # noqa: F401
    _is_valid_ip,
    preprocess,
)
from sld.features import (  # noqa: F401
    _distinct_in_window,
    _window_counts,
    build_features,
    haversine_km,
)
from sld.models import (  # noqa: F401
    ML_EXTRA,
    ML_FEATURES,
    SUP_FEATURES,
    _SUP_CATEGORICAL,
    _make_hgb,
    add_ml_scores,
    add_supervised_scores,
)
from sld.scoring import (  # noqa: F401
    MODEL_COLS,
    RULE_COLS,
    build_incidents,
    score_events,
)
from sld.storage import (  # noqa: F401
    HASH_FIELDS,
    STATUSES,
    mask_ip,
    record_hash,
    save_to_db,
    update_status,
    verify_integrity,
)
from sld.evaluation import (  # noqa: F401
    _split_by_user,
    confusion_metrics,
    evaluate_events,
    evaluate_incidents,
    evaluate_rba_events,
    evaluate_rba_rules,
    evaluate_rba_summary,
    sensitivity_sweep,
)
from sld.rba_loader import (  # noqa: F401
    RBA_USECOLS,
    _CSV_KW,
    _flag,
    _strip_version,
    load_rba_subset,
)
from sld.pipeline import (  # noqa: F401
    _prepare_db_tables,
    main,
    run_pipeline,
)


if __name__ == "__main__":
    main()
