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

# -------------------- Config par défaut --------------------
DEFAULT_STEP_MIN = 10
DEFAULT_LABEL = "y48"  # le CSV historique doit avoir y48 (et/ou y72)
DEFAULT_PTHR = 0.55    # seuil proba pour alerte
DEFAULT_PWIN = 6       # persistance: fenêtre (6 x 10min ≈ 1h)
DEFAULT_PNEED = 3      # persistance: au moins 3/6
MODEL_DIR = Path("model"); MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "model.pkl"

# -------------------- Helpers / Features --------------------
def ema(s, span): 
    return s.ewm(span=span, adjust=False).mean()

def build_features(df: pd.DataFrame, step_minutes: int = DEFAULT_STEP_MIN) -> pd.DataFrame:
    """Features physiques compactes par transfo."""
    out = []
    fac = 60 / step_minutes
    for _, g in df.sort_values(["tx_id", "ts"]).groupby("tx_id", sort=False):
        g = g.copy()
        g["dT_dt"] = g["T_wdg"].diff() * fac
        g["dT_elev"] = g["T_wdg"] - g["T_amb"]
        g["load_ema"] = ema(g["load"], 12)
        g["thermal_margin"] = 115.0 - g["T_wdg"]
        out.append(g)
    feats = pd.concat(out, axis=0)
    return feats

def persistent_alert(series_bool: pd.Series, window: int, at_least: int) -> pd.Series:
    """Alerte si au moins `at_least` points positifs dans une fenêtre glissante de taille `window`."""
    return series_bool.rolling(window=window, min_periods=1).sum() >= at_least

# -------------------- Learn / Predict --------------------
def learn_model(df_hist: pd.DataFrame, label_col: str, step_minutes: int):
    """Entraîne RF sur l'historique et sauve model.pkl"""
    if label_col not in df_hist.columns:
        raise ValueError(f"Le label '{label_col}' est absent du CSV historique.")
    feats = build_features(df_hist, step_minutes)
    feats = feats.merge(df_hist[["ts", "tx_id", label_col]], on=["ts", "tx_id"], how="left")

    X = feats[["dT_dt", "dT_elev", "load_ema", "thermal_margin"]].fillna(0.0)
    y = feats[label_col].astype(int)
    strat = y if 0 < y.sum() < len(y) else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=strat)

    clf = RandomForestClassifier(
        n_estimators=300, n_jobs=-1, class_weight="balanced_subsample", random_state=42
    )
    clf.fit(Xtr, ytr)
    pte = clf.predict_proba(Xte)[:, 1]
    auc = float("nan") if len(np.unique(yte)) == 1 else roc_auc_score(yte, pte)
    rep = classification_report(yte, (pte >= 0.5).astype(int), output_dict=True, zero_division=0)

    joblib.dump({"model": clf, "step": step_minutes, "label": label_col}, MODEL_PATH)
    return auc, rep

def predict_stream(df_rt: pd.DataFrame, pthr: float, pwin: int, pneed: int, model_path: Path = MODEL_PATH):
    """Charge le modèle, score le flux, calcule alertes & lead times si possible."""
    bundle = joblib.load(model_path)
    clf = bundle["model"]; step = int(bundle.get("step", DEFAULT_STEP_MIN))

    feats = build_features(df_rt, step)
    X = feats[["dT_dt", "dT_elev", "load_ema", "thermal_margin"]].fillna(0.0)
    feats["risk"] = clf.predict_proba(X)[:, 1]
    feats["alert_raw"] = feats["risk"] >= pthr

    feats["alert"] = False
    for tx, g in feats.groupby("tx_id"):
        m = feats["tx_id"] == tx
        feats.loc[m, "alert"] = persistent_alert(g["alert_raw"], pwin, pneed).values

    # Lead time si failure_time est présent (optionnel)
    lead_df = pd.DataFrame()
    if "failure_time" in df_rt.columns:
        rows = []
        for tx, g in feats.groupby("tx_id"):
            t_fail_vals = df_rt.loc[df_rt["tx_id"] == tx, "failure_time"].dropna().unique()
            if len(t_fail_vals) == 0:
                continue
            t_fail = pd.to_datetime(t_fail_vals[0])
            g = g.sort_values("ts")
            before = g[g["ts"] < t_fail]
            first = before.loc[before["alert"], "ts"].min() if not before.loc[before["alert"], "ts"].empty else pd.NaT
            lead_h = (t_fail - first).total_seconds() / 3600.0 if pd.notna(first) else 0.0
            rows.append({"tx_id": tx, "failure_time": t_fail, "lead_h": lead_h})
        lead_df = pd.DataFrame(rows)

    # Joindre T amb/wdg pour export
    feats = feats.merge(df_rt[["ts", "tx_id", "T_amb", "T_wdg"]], on=["ts", "tx_id"], how="left")
    feats = feats.sort_values(["tx_id", "ts"])

    return feats, lead_df

# -------------------- UI --------------------
st.set_page_config(page_title="SmartPredict — Learn & Predict", layout="wide")
st.title("SmartPredict — IA (Learn / Predict)")

with st.expander("⚙️ Paramètres", expanded=True):
    colp1, colp2, colp3, colp4 = st.columns(4)
    label_col = colp1.selectbox("Label historique", ["y48", "y72"], index=0)
    step_minutes = colp2.number_input("Pas de temps (minutes)", 5, 60, DEFAULT_STEP_MIN, step=5)
    pthr = colp3.slider("Seuil proba (alerte)", 0.05, 0.95, DEFAULT_PTHR, 0.05)
    pwin = colp4.number_input("Fenêtre persistance (#points)", 1, 60, DEFAULT_PWIN)
    pneed = st.number_input("Au moins (#positifs dans la fenêtre)", 1, 60, DEFAULT_PNEED)

st.markdown("—")

tab1, tab2 = st.tabs(["📚 LEARN (historique)", "🔮 PREDICT (temps réel)"])

# ============ LEARN ============
with tab1:
    st.subheader("Entraîner l'IA sur l'historique")
    hist_file = st.file_uploader("Importer le CSV historique (avec colonnes ts, tx_id, T_amb, load, T_wdg, y48…)", type=["csv"], key="hist")
    if st.button("LEARN", type="primary", use_container_width=True, disabled=hist_file is None):
        try:
            df_hist = pd.read_csv(hist_file, parse_dates=["ts"])
            auc, rep = learn_model(df_hist, label_col, int(step_minutes))
            st.success(f"✅ Modèle entraîné et sauvegardé → {MODEL_PATH}")
            st.write(f"ROC AUC = **{auc:.3f}**")
            st.json({
                "precision": round(rep["weighted avg"]["precision"], 3),
                "recall":    round(rep["weighted avg"]["recall"], 3),
                "f1":        round(rep["weighted avg"]["f1-score"], 3)
            })
        except Exception as e:
            st.error(f"Erreur LEARN : {e}")

# ============ PREDICT ============
with tab2:
    st.subheader("Prédire sur un flux temps réel (inconnu)")
    rt_file = st.file_uploader("Importer le CSV temps réel (ts, tx_id, T_amb, load, T_wdg, [failure_time optionnel])", type=["csv"], key="rt")
    colb1, colb2 = st.columns(2)
    do_plot = colb1.checkbox("Afficher les courbes", value=True)
    chosen_tx = colb2.number_input("TX pour visualisation (si courbes activées)", min_value=1, value=1, step=1)

    if st.button("PREDICT", type="primary", use_container_width=True, disabled=rt_file is None):
        try:
            df_rt = pd.read_csv(rt_file, parse_dates=["ts"])
            feats_scored, lead_df = predict_stream(df_rt, float(pthr), int(pwin), int(pneed), MODEL_PATH)
            st.success("✅ Prédiction effectuée")

            # Téléchargement CSV
            cols = ["ts","tx_id","T_amb","T_wdg","dT_dt","dT_elev","load_ema","thermal_margin","risk","alert"]
            export_cols = [c for c in cols if c in feats_scored.columns]
            export_df = feats_scored[export_cols].copy()

            buf = io.BytesIO()
            export_df.to_csv(buf, index=False); buf.seek(0)
            st.download_button("⬇️ Télécharger features_scored.csv", data=buf, file_name="features_scored.csv", mime="text/csv")

            # Lead times
            if not lead_df.empty:
                st.markdown("#### Lead times (heures)")
                st.dataframe(lead_df.sort_values("tx_id"))

            # Courbes
            if do_plot:
                g = feats_scored[feats_scored["tx_id"] == int(chosen_tx)].sort_values("ts")
                if not g.empty:
                    fig, ax1 = plt.subplots(figsize=(10, 4))
                    ax1.plot(g["ts"], g["T_wdg"], label="T_wdg (°C)")
                    ax1.set_ylabel("Température (°C)")
                    ax2 = ax1.twinx()
                    ax2.plot(g["ts"], g["risk"], label="Risk", alpha=0.7)
                    ax2.axhline(pthr, ls="--", alpha=0.4)
                    ax2.set_ylabel("Probabilité de panne")
                    ax1.set_title(f"TX {int(chosen_tx)} — Température & Risk")
                    ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
                    fig.autofmt_xdate(); st.pyplot(fig)
                else:
                    st.info("Aucune donnée pour ce TX dans le fichier.")
        except FileNotFoundError:
            st.error("Aucun modèle trouvé. Lance d'abord LEARN pour créer model/model.pkl.")
        except Exception as e:
            st.error(f"Erreur PREDICT : {e}")
