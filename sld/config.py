"""Settings (CONFIG thresholds) and country coordinates.  (split from detector.py)"""



# ---- config ----
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
