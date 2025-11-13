#!/usr/bin/env python
"""
Génère un dataset "temps réel" compatible avec l'historique.
Entrées (args): --n_tx, --days, --step, --seed, --fail_tx (option), --out
Si --fail_tx est fourni, ce transfo subit une dérive progressive vers panne;
sinon aucun transfo ne tombe en panne.
Sortie: CSV (realtime.csv)
"""
from pathlib import Path
import numpy as np, pandas as pd

def _apply_phase_ramp(T_wdg, start_idx, end_idx, slope_per_step):
    span = max(1, end_idx - start_idx)
    ramp = np.arange(span) * slope_per_step
    T_wdg[start_idx:end_idx] += ramp
    last_inc = ramp[-1] if span>0 else 0.0
    if end_idx < len(T_wdg): T_wdg[end_idx:] += last_inc

def simulate_fleet(n_tx=10, days=45, step_minutes=10, seed=777, fail_tx=1):
    rng = np.random.default_rng(seed)
    spd = (24*60)//step_minutes
    n = spd*days
    ts = pd.date_range("2025-03-01", periods=n, freq=f"{step_minutes}min")
    rows=[]
    for tx in range(1, n_tx+1):
        T_amb = 35 + 4*np.sin(2*np.pi*np.arange(n)/spd) + rng.normal(0,0.5,n)
        load  = 30 + 50*np.maximum(0, np.sin(2*np.pi*(np.arange(n)-0.25*spd)/spd)) + rng.normal(0,4,n)
        load  = np.clip(load, 10, 100)
        tgt   = T_amb + 40 + 25*(load/100.0)**1.2
        alpha = 0.9
        T_wdg = np.empty(n); T_wdg[0]=tgt[0]
        for i in range(1,n): T_wdg[i] = alpha*T_wdg[i-1] + (1-alpha)*tgt[i]

        if (fail_tx is not None) and (tx==fail_tx):
            # dérive sur les derniers jours (réaliste)
            end_day = days - 0.5
            start_day = max(1.0, end_day-6.0)
            phases = [
                {"start_day":start_day, "end_day":end_day-1.0, "slope_c_per_h":0.04},
                {"start_day":end_day-1.0, "end_day":end_day,  "slope_c_per_h":1.0},
            ]
            for ph in phases:
                a,b = int(ph["start_day"]*spd), int(ph["end_day"]*spd)
                a = max(0, min(a, n-2)); b = max(a+1, min(b, n-1))
                slope = ph["slope_c_per_h"]*(step_minutes/60.0)
                _apply_phase_ramp(T_wdg, a, b, slope)

        for i in range(n): rows.append([ts[i], tx, T_amb[i], load[i], T_wdg[i]])

    df = pd.DataFrame(rows, columns=["ts","tx_id","T_amb","load","T_wdg"])
    df["ts"]=pd.to_datetime(df["ts"])
    return df

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_tx", type=int, default=10)
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--step", type=int, default=10)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--fail_tx", type=int, default=None, help="tx_id à faire tomber en panne")
    p.add_argument("--out", type=str, default="data_realtime")
    args = p.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    df = simulate_fleet(args.n_tx, args.days, args.step, args.seed, args.fail_tx)
    df.to_csv(outdir/"realtime.csv", index=False)
    print(f"[OK] {outdir/'realtime.csv'}")

if __name__ == "__main__":
    main()
