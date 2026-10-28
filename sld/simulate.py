"""Simulated login logs (the Milestone-3 demo data).  (split from detector.py)"""

import numpy as np
import pandas as pd
from sld.config import COUNTRY_INFO


# ---- generate ----
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
