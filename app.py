import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib

# -------------------- Config --------------------
DEFAULT_STEP_MIN = 10
DEFAULT_LABEL = "y48"
DEFAULT_PTHR = 0.55
DEFAULT_PWIN = 6
DEFAULT_PNEED = 3
T_CRIT = 115.0  # température critique de référence
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "model.pkl"

# -------------------- Helpers --------------------
def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def build_features(df: pd.DataFrame, step_minutes: int = DEFAULT_STEP_MIN) -> pd.DataFrame:
    """
    Construit des features physiques simples à partir du CSV brut.
    Le dataframe d'entrée doit déjà être trié par (tx_id, ts).
    """
    out = []
    fac = 60 / step_minutes
    for _, g in df.groupby("tx_id", sort=False):
        g = g.copy()
        g["dT_dt"] = g["T_wdg"].diff() * fac          # dérivée °C/h
        g["dT_elev"] = g["T_wdg"] - g["T_amb"]        # surélévation thermique
        g["load_ema"] = ema(g["load"], 12)            # charge lissée
        g["thermal_margin"] = T_CRIT - g["T_wdg"]     # marge avant T_CRIT (115 - T_wdg)
        out.append(g)
    return pd.concat(out, axis=0)

def persistent_alert(series_bool: pd.Series, window: int, at_least: int) -> pd.Series:
    """Alerte = au moins `at_least` True dans une fenêtre glissante de taille `window`."""
    return series_bool.rolling(window=window, min_periods=1).sum() >= at_least

# -------------------- IA : LEARN --------------------
def learn_model(df_hist: pd.DataFrame, label_col: str, step_minutes: int):
    """
    Entraîne le modèle à partir du dataset historique figé.
    Le label (y48 ou y72) doit déjà exister dans df_hist.
    On ne modifie pas df_hist, on le trie juste en mémoire pour aligner X et y.
    """
    if label_col not in df_hist.columns:
        raise ValueError(
            f"Le label '{label_col}' est absent du CSV historique.\n"
            f"Colonnes disponibles : {list(df_hist.columns)}"
        )

    # 1) On travaille sur une copie triée pour garantir l'alignement X/y
    df_sorted = df_hist.sort_values(["tx_id", "ts"]).reset_index(drop=True)

    # 2) Features physiques à partir du dataframe trié
    feats = build_features(df_sorted, step_minutes)

    # 3) X = features, y = colonne label du dataset historique figé
    X = feats[["dT_dt", "dT_elev", "load_ema", "thermal_margin"]].fillna(0.0)
    y = df_sorted[label_col].astype(int)

    strat = y if 0 < y.sum() < len(y) else None

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=strat
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42,
    )
    clf.fit(Xtr, ytr)
    pte = clf.predict_proba(Xte)[:, 1]
    auc = float("nan") if len(np.unique(yte)) == 1 else roc_auc_score(yte, pte)
    rep = classification_report(
        yte, (pte >= 0.5).astype(int), output_dict=True, zero_division=0
    )

    joblib.dump({"model": clf, "step": step_minutes, "label": label_col}, MODEL_PATH)
    return auc, rep

# -------------------- IA : PREDICT --------------------
def predict_stream(df_rt: pd.DataFrame, pthr: float, pwin: int, pneed: int):
    """
    Applique le modèle entraîné sur un flux temps réel.
    Calcule risk, alert, et lead time (si failure_time existe).
    Garantit que T_amb/T_wdg sont ramenés si présents dans le CSV temps réel.
    """
    bundle = joblib.load(MODEL_PATH)
    clf = bundle["model"]
    step = int(bundle.get("step", DEFAULT_STEP_MIN))

    # Tri pour cohérence
    df_sorted = df_rt.sort_values(["tx_id", "ts"]).reset_index(drop=True)
    feats = build_features(df_sorted, step)
    X = feats[["dT_dt", "dT_elev", "load_ema", "thermal_margin"]].fillna(0.0)

    feats["risk"] = clf.predict_proba(X)[:, 1]
    feats["alert_raw"] = feats["risk"] >= pthr

    feats["alert"] = False
    for tx, g in feats.groupby("tx_id"):
        m = feats["tx_id"] == tx
        feats.loc[m, "alert"] = persistent_alert(g["alert_raw"], pwin, pneed).values

    # Lead time si failure_time présent
    lead_df = pd.DataFrame()
    if "failure_time" in df_sorted.columns:
        rows = []
        for tx, g in feats.groupby("tx_id"):
            t_vals = df_sorted.loc[df_sorted["tx_id"] == tx, "failure_time"].dropna().unique()
            if len(t_vals) == 0:
                continue
            t_fail = pd.to_datetime(t_vals[0])
            g = g.sort_values("ts")
            before = g[g["ts"] < t_fail]
            first = before.loc[before["alert"], "ts"].min() if not before.loc[before["alert"], "ts"].empty else pd.NaT
            lead_h = (t_fail - first).total_seconds() / 3600.0 if pd.notna(first) else 0.0
            rows.append({"tx_id": tx, "failure_time": t_fail, "lead_h": lead_h})
        lead_df = pd.DataFrame(rows)

    # Merge T_amb / T_wdg pour export & plots, uniquement si elles existent dans le CSV
    merge_cols = ["ts", "tx_id"]
    extra_cols = []
    for col in ["T_amb", "T_wdg"]:
        if col in df_sorted.columns:
            extra_cols.append(col)
    if extra_cols:
        feats = feats.merge(
            df_sorted[merge_cols + extra_cols],
            on=["ts", "tx_id"],
            how="left",
        )

    feats = feats.sort_values(["tx_id", "ts"])
    return feats, lead_df

# -------------------- UI Streamlit --------------------
st.set_page_config(page_title="SmartPredict — IA Transfo", layout="wide")
st.title("⚡ SmartPredict — Learn / Predict")

with st.expander("⚙️ Paramètres IA", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    label_col = c1.selectbox("Label historique", ["y48", "y72"], index=0)
    step_minutes = c2.number_input("Pas de temps (minutes)", 5, 60, DEFAULT_STEP_MIN, step=5)
    pthr = c3.slider("Seuil probabilité (alerte)", 0.05, 0.95, DEFAULT_PTHR, 0.05)
    pwin = c4.number_input("Fenêtre persistance (#points)", 1, 60, DEFAULT_PWIN)
    pneed = st.number_input("Min. points positifs dans la fenêtre", 1, 60, DEFAULT_PNEED)

st.markdown("---")

tab1, tab2 = st.tabs(["📚 LEARN (historique)", "🔮 PREDICT (temps réel)"])

# ---------- Onglet LEARN ----------
with tab1:
    st.subheader("Entraîner l'algorithme sur l'historique (dataset figé)")
    hist_file = st.file_uploader(
        "CSV historique (ts, tx_id, T_amb, load, T_wdg, y48, y72, ...)",
        type=["csv"],
        key="hist",
    )

    if st.button("LEARN", type="primary", use_container_width=True, disabled=hist_file is None):
        try:
            # 1) Lecture brute
            df_hist = pd.read_csv(hist_file)

            # 2) Normalisation des noms de colonnes (uniquement en mémoire)
            df_hist.columns = [c.strip() for c in df_hist.columns]

            # 3) Conversion de ts si présent
            if "ts" in df_hist.columns:
                df_hist["ts"] = pd.to_datetime(df_hist["ts"])

            # 4) Debug visible
            st.write("Colonnes détectées dans le CSV historique :", list(df_hist.columns))

            # 5) Label nettoyé (au cas où il y aurait des espaces dans le selectbox)
            cleaned_label = label_col.strip()

            # 6) Entraînement
            auc, rep = learn_model(df_hist, cleaned_label, int(step_minutes))
            st.success(f"✅ Modèle entraîné et sauvegardé → {MODEL_PATH}")
            st.write(f"**ROC AUC = {auc:.3f}**")
            st.json(
                {
                    "precision": round(rep["weighted avg"]["precision"], 3),
                    "recall": round(rep["weighted avg"]["recall"], 3),
                    "f1": round(rep["weighted avg"]["f1-score"], 3),
                }
            )
        except Exception as e:
            st.error(f"Erreur LEARN : {e}")

# ---------- Onglet PREDICT ----------
with tab2:
    st.subheader("Prédire sur un flux temps réel (inconnu)")
    rt_file = st.file_uploader(
        "CSV temps réel (ts, tx_id, T_amb, load, T_wdg, [failure_time optionnel])",
        type=["csv"],
        key="rt",
    )
    c5, c6 = st.columns(2)
    do_plot = c5.checkbox("Afficher les courbes pour un transfo", value=True)
    tx_plot = int(c6.number_input("TX à visualiser", min_value=1, value=1, step=1))

    if st.button("PREDICT", type="primary", use_container_width=True, disabled=rt_file is None):
        try:
            # 1) Lecture brute
            df_rt = pd.read_csv(rt_file)

            # 2) Normalisation des colonnes
            df_rt.columns = [c.strip() for c in df_rt.columns]

            # 3) Conversion ts si présent
            if "ts" in df_rt.columns:
                df_rt["ts"] = pd.to_datetime(df_rt["ts"])

            st.write("Colonnes détectées dans le CSV temps réel :", list(df_rt.columns))

            # 4) Prédiction
            feats_scored, lead_df = predict_stream(df_rt, float(pthr), int(pwin), int(pneed))
            st.success("✅ Prédiction effectuée")

            # 5) Export CSV pour PowerBI
            export_cols = [
                "ts", "tx_id", "T_amb", "T_wdg",
                "dT_dt", "dT_elev", "load_ema", "thermal_margin",
                "risk", "alert",
            ]
            export_cols = [c for c in export_cols if c in feats_scored.columns]
            export_df = feats_scored[export_cols].copy()

            buf = io.BytesIO()
            export_df.to_csv(buf, index=False)
            buf.seek(0)
            st.download_button(
                "⬇️ Télécharger features_scored.csv",
                data=buf,
                file_name="features_scored.csv",
                mime="text/csv",
            )

            # 6) Lead times
            if not lead_df.empty:
                st.markdown("### ⏱ Lead time (heures) par transfo")
                st.dataframe(lead_df.sort_values("tx_id"))

            # 7) Courbes pour un transfo
            if do_plot:
                g = feats_scored[feats_scored["tx_id"] == tx_plot].sort_values("ts")
                if not g.empty:
                    # Récupérer la panne & la première alerte pour ce TX si possible
                    t_fail = None
                    if "failure_time" in df_rt.columns:
                        t_vals = df_rt.loc[df_rt["tx_id"] == tx_plot, "failure_time"].dropna().unique()
                        if len(t_vals) > 0:
                            t_fail = pd.to_datetime(t_vals[0])

                    first_alert_ts = None
                    if t_fail is not None and "alert" in g.columns:
                        before = g[g["ts"] < t_fail]
                        if not before.loc[before["alert"], "ts"].empty:
                            first_alert_ts = before.loc[before["alert"], "ts"].min()

                    # ---- Graphique 1 : probabilité de panne ----
                    fig1, ax1 = plt.subplots(figsize=(10, 4))
                    ax1.plot(g["ts"], g["risk"], label="Risk IA")
                    ax1.axhline(pthr, ls="--", alpha=0.6, label=f"Seuil {pthr:.2f}")
                    if first_alert_ts is not None:
                        ax1.axvline(first_alert_ts, color="orange", ls="--", label="Première alerte")
                    if t_fail is not None:
                        ax1.axvline(t_fail, color="red", ls="-", label="Panne")
                    ax1.set_ylabel("Probabilité de panne")
                    ax1.set_title(f"TX {tx_plot} — Probabilité de panne (risk) vs seuil")
                    ax1.legend(loc="upper left")
                    fig1.autofmt_xdate()
                    st.pyplot(fig1)

                    # ---- Graphique 2 : température & marge thermique (115 - T_wdg) ----
                    fig2, axT = plt.subplots(figsize=(10, 4))

                    # Température
                    if "T_wdg" in g.columns:
                        axT.plot(g["ts"], g["T_wdg"], label="T_wdg (°C)")

                    # Marge thermique (115 - T_wdg) : soit prise depuis thermal_margin, soit calculée à la volée
                    if "thermal_margin" in g.columns:
                        margin = g["thermal_margin"]
                    elif "T_wdg" in g.columns:
                        margin = T_CRIT - g["T_wdg"]
                    else:
                        margin = None

                    if margin is not None:
                        axT.plot(g["ts"], margin, label="Marge thermique (°C)")

                    if first_alert_ts is not None:
                        axT.axvline(first_alert_ts, color="orange", ls="--", label="Première alerte")
                    if t_fail is not None:
                        axT.axvline(t_fail, color="red", ls="-", label="Panne")

                    axT.set_ylabel("Température / Marge (°C)")
                    axT.set_title(
                        f"TX {tx_plot} — Température et distance au seuil critique (115°C)"
                    )
                    axT.legend(loc="upper left")
                    fig2.autofmt_xdate()
                    st.pyplot(fig2)
                else:
                    st.info(f"Aucune donnée pour tx_id={tx_plot} dans ce fichier.")
        except FileNotFoundError:
            st.error("Aucun modèle trouvé. Lance d'abord LEARN pour créer model/model.pkl.")
        except Exception as e:
            st.error(f"Erreur PREDICT : {e}")
