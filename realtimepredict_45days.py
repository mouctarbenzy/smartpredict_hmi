#!/usr/bin/env python
"""
Génère un dataset "temps réel" compatible avec l'historique.

- 45 jours par défaut
- 10 transformateurs
- 2 transformateurs en panne choisis aléatoirement
- Pour chaque transfo en panne :
    * une date de panne aléatoire (avec au moins 7 jours d'historique avant)
    * une dérive thermique linéaire sur 7 jours (+20°C au total)

Aucun graphique ici : ce script ne fait que générer un CSV.

Sortie : realtime.csv dans le dossier data_realtime/
Colonnes :
    ts, tx_id, T_amb, load, T_wdg, failure_time
"""

from pathlib import Path
import numpy as np
import pandas as pd


def simulate_fleet(
    n_tx: int = 10,
    days: int = 45,
    step_minutes: int = 10,
    seed: int | None = None,
    n_faults: int = 2,
) -> tuple[pd.DataFrame, dict]:
    """
    Simule une flotte de transformateurs.

    Retourne :
      - df : DataFrame avec les colonnes [ts, tx_id, T_amb, load, T_wdg, failure_time]
      - fault_info : dict {tx_id: failure_time}
    """
    # RNG : aléatoire vrai si seed=None, reproductible si seed fixé
    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    steps_per_day = (24 * 60) // step_minutes
    n_steps = steps_per_day * days
    ts = pd.date_range("2025-03-01", periods=n_steps, freq=f"{step_minutes}min")

    # Choix aléatoire des transformateurs en panne
    all_txs = np.arange(1, n_tx + 1)
    n_faults = min(n_faults, n_tx)
    fault_txs = rng.choice(all_txs, size=n_faults, replace=False)

    rows = []
    fault_info: dict[int, pd.Timestamp] = {}

    for tx in all_txs:
        # Température ambiante jour/nuit + bruit léger
        idx = np.arange(n_steps)
        T_amb = 35 + 4 * np.sin(2 * np.pi * idx / steps_per_day) + rng.normal(
            0, 0.5, n_steps
        )

        # Charge journalière + bruit
        load = (
            30
            + 50
            * np.maximum(
                0,
                np.sin(
                    2 * np.pi * (idx - 0.25 * steps_per_day) / steps_per_day
                ),
            )
            + rng.normal(0, 4, n_steps)
        )
        load = np.clip(load, 10, 100)

        # Cible thermique sans dérive
        tgt = T_amb + 40 + 25 * (load / 100.0) ** 1.2

        # Dynamique thermique de base (filtre 1er ordre)
        alpha = 0.9
        T_wdg = np.empty(n_steps)
        T_wdg[0] = tgt[0]
        for i in range(1, n_steps):
            T_wdg[i] = alpha * T_wdg[i - 1] + (1 - alpha) * tgt[i]

        # Par défaut, pas de panne
        failure_time = pd.NaT

        if tx in fault_txs:
            # On impose au moins 7 jours d'historique avant la panne
            min_fail_step = 7 * steps_per_day
            max_fail_step = n_steps - 1
            if min_fail_step >= max_fail_step:
                fail_step = max_fail_step
            else:
                fail_step = int(rng.integers(min_fail_step, max_fail_step + 1))

            # Dérive sur 7 jours = 7 * steps_per_day pas
            drift_start_step = fail_step - 7 * steps_per_day
            drift_start_step = max(0, drift_start_step)
            span = max(1, fail_step - drift_start_step)

            total_drift = 20.0  # +20°C au moment de la panne
            slope_per_step = total_drift / span

            ramp = np.arange(span) * slope_per_step
            T_wdg[drift_start_step:fail_step] += ramp
            # Après la panne, on fige la température (transfo en défaut)
            T_wdg[fail_step:] = T_wdg[fail_step]

            failure_time = ts[fail_step]
            fault_info[tx] = failure_time

        # Ajout des lignes
        for i in range(n_steps):
            rows.append(
                [
                    ts[i],
                    tx,
                    T_amb[i],
                    load[i],
                    T_wdg[i],
                    failure_time,
                ]
            )

    df = pd.DataFrame(
        rows,
        columns=["ts", "tx_id", "T_amb", "load", "T_wdg", "failure_time"],
    )
    df["ts"] = pd.to_datetime(df["ts"])
    return df, fault_info


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n_tx", type=int, default=10, help="Nombre de transformateurs")
    p.add_argument("--days", type=int, default=45, help="Nombre de jours à simuler")
    p.add_argument("--step", type=int, default=10, help="Pas de temps (minutes)")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Graine aléatoire (None = vraiment aléatoire à chaque run)",
    )
    p.add_argument(
        "--n_faults",
        type=int,
        default=2,
        help="Nombre de transformateurs en panne",
    )
    p.add_argument(
        "--out",
        type=str,
        default="data_realtime",
        help="Dossier de sortie pour realtime.csv",
    )
    args = p.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    df, fault_info = simulate_fleet(
        n_tx=args.n_tx,
        days=args.days,
        step_minutes=args.step,
        seed=args.seed,
        n_faults=args.n_faults,
    )

    out_path = outdir / "realtime.csv"
    df.to_csv(out_path, index=False)

    print(f"\n[OK] Dataset généré → {out_path}")
    print(f"Lignes : {len(df)}, Colonnes : {list(df.columns)}\n")

    if fault_info:
        print("=== TRANSFORMATEURS EN PANNE ===")
        print(f"Liste : {list(fault_info.keys())}\n")
        for tx, ft in fault_info.items():
            print(f" - TX{tx}  →  panne le {ft}")
    else:
        print("[INFO] Aucun transfo en panne.\n")


if __name__ == "__main__":
    main()
