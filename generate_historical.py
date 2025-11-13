#!/usr/bin/env python
"""
Génère un dataset historique "figé" avec dérives/probables pannes.
Entrées (args): --n_tx, --days, --n_fail, --step, --seed, --out
Sorties: CSV (historical.csv) + JSON (historical_meta.json)
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

def _apply_phase_ramp(T_wdg, start_idx, end_idx, slope_per_step):
    span = max(1, end_idx - start_idx)
    ramp = np.arange(span) * slope_per_step
    T_wdg[start_idx:end_idx] += ramp
    last_inc = ramp[-1] if span>0 else 0.0
    if end_idx < len(T_wdg): T_wdg[end_idx:] += last_inc

def simulate_fleet(n_tx=20, days=60, step_minutes=10, seed=123, failures=None):
    rng = np.random.default_rng(seed)
    spd = (24*60)//step_minutes
    n = spd*days
    ts = pd.date_range("2025-01-01", periods=n, freq=f"{step_minutes}min")
    rows=[]; meta={"failures":[]}

    for tx in range(1, n_tx+1):
        T_amb = 35 + 4*np.sin(2*np.pi*np.arange(n)/spd) + rng.normal(0,0.5,n)
        load  = 30 + 50*np.maximum(0, np.sin(2*np.pi*(np.arange(n)-0.25*spd)/spd)) + rng.normal(0,4,n)
        load  = np.clip(load, 10, 100)
        tgt   = T_amb + 40 + 25*(load/100.0)**1.2
        alpha = 0.9
        T_wdg = np.empty(n); T_wdg[0]=tgt[0]
        for i in range(1, n): T_wdg[i] = alpha*T_wdg[i-1] + (1-alpha)*tgt[i]

        # éventuelle panne (multi-phases réaliste)
        if failures and any(f["tx_id"]==tx for f in failures):
            f = [f for f in failures if f["tx_id"]==tx][0]
            for ph in f["phases"]:
                a,b = int(ph["start_day"]*spd), int(ph["end_day"]*spd)
                a = max(0, min(a, n-2)); b = max(a+1, min(b, n-1))
                slope = ph["slope_c_per_h"]*(step_minutes/60.0)
                _apply_phase_ramp(T_wdg, a, b, slope)
            meta["failures"].append({
                "tx_id": tx,
                "failure_day": f["phases"][-1]["end_day"]
            })

        for i in range(n): rows.append([ts[i], tx, T_amb[i], load[i], T_wdg[i]])

    df = pd.DataFrame(rows, columns=["ts","tx_id","T_amb","load","T_wdg"])
    df["ts"] = pd.to_datetime(df["ts"])
    return df, meta

def random_failures(n_tx, n_fail, days, rng):
    # tire n_fail transfos distincts + moments de panne
    tx_ids = rng.choice(np.arange(1, n_tx+1), size=min(n_fail, n_tx), replace=False)
    fails=[]
    for tx in tx_ids:
        end_day = rng.uniform(low=0.6*days, high=0.95*days)     # panne dans le dernier tiers
        start_day = max(1.0, end_day - rng.uniform(1.5, 10.0))  # dérive commence 1.5–10 j avant
        # deux phases: lente + accélération finale
        phases = [
            {"start_day": start_day, "end_day": max(start_day+1.0, end_day-1.0), "slope_c_per_h": 0.03},
            {"start_day": max(start_day+1.0, end_day-1.0), "end_day": end_day,  "slope_c_per_h": 0.9},
        ]
        fails.append({"tx_id": int(tx), "phases": phases})
    return fails

def make_labels(df, meta, step_minutes=10, horizons=(48,72)):
    spd = (24*60)//step_minutes
    df = df.copy()
    df["failure_time"] = pd.NaT
    for H in horizons: df[f"y{H}"] = 0
    all_ts = df["ts"].sort_values().unique()
    for f in meta["failures"]:
        end_day = f["failure_day"]
        fail_idx = min(int(end_day*spd), len(all_ts)-1)
        t_fail = pd.to_datetime(all_ts[fail_idx])
        m = df["tx_id"]==f["tx_id"]
        df.loc[m, "failure_time"] = t_fail
        for H in horizons:
            df.loc[m & (df["ts"]>=t_fail-pd.Timedelta(hours=H)) & (df["ts"]<t_fail), f"y{H}"] = 1
    return df

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_tx", type=int, default=20)
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--n_fail", type=int, default=3)
    p.add_argument("--step", type=int, default=10, help="minutes")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", type=str, default="data_historical")
    args = p.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    fails = random_failures(args.n_tx, args.n_fail, args.days, rng)
    df, meta = simulate_fleet(args.n_tx, args.days, args.step, args.seed, failures=fails)
    df = make_labels(df, meta, step_minutes=args.step, horizons=(48,72))

    df.to_csv(outdir/"historical.csv", index=False)
    (outdir/"historical_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] {outdir/'historical.csv'}  (+ {outdir/'historical_meta.json'})")

if __name__ == "__main__":
    main()
