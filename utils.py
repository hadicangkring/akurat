# eval_utils.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
import os

# === BACA LOG ===
def load_log(path="data/prediksi_log.csv"):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if "prediksi_4digit" in df.columns and "real_4digit" in df.columns:
            return df
    except Exception:
        pass
    return pd.DataFrame()

# === METRIK UTAMA ===
def exact_match_rate(df):
    if df.empty: return 0
    return (df['prediksi_4digit'].astype(str) == df['real_4digit'].astype(str)).mean()

def per_position_accuracy(df):
    if df.empty: return {}
    acc = {}
    for pos in range(4):
        correct = df.apply(lambda r: str(r['prediksi_4digit']).zfill(4)[pos] == str(r['real_4digit']).zfill(4)[pos], axis=1)
        acc[f'pos{pos+1}'] = correct.mean()
    return acc

def avg_similarity(df):
    if df.empty: return 0
    return df.apply(lambda r: SequenceMatcher(None, str(r['prediksi_4digit']).zfill(4), str(r['real_4digit']).zfill(4)).ratio(), axis=1).mean()

# === GRAFIK TREN ===
def plot_rolling_accuracy(df, window=20):
    if df.empty:
        return None
    df = df.copy()
    df["correct"] = (df["prediksi_4digit"].astype(str) == df["real_4digit"].astype(str)).astype(int)
    df["rolling_acc"] = df["correct"].rolling(window=window, min_periods=5).mean()

    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(df.index, df["rolling_acc"], label=f"Rolling {window}-window")
    ax.set_title("Tren Akurasi Prediksi (Top-1)")
    ax.set_xlabel("Prediksi ke-")
    ax.set_ylabel("Akurasi")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig

# === FUNGSI UTAMA UNTUK STREAMLIT ===
def tampilkan_evaluasi(st, log_path="data/prediksi_log.csv"):
    st.subheader("📈 Evaluasi Kinerja Model Historis")

    df = load_log(log_path)
    if df.empty:
        st.info("Belum ada data prediksi yang tersimpan untuk evaluasi.")
        return

    exact = exact_match_rate(df)
    perpos = per_position_accuracy(df)
    sim = avg_similarity(df)

    st.markdown(f"**🎯 Top-1 Exact Match:** {exact:.2%}")
    st.markdown(f"**🧩 Rata-rata Kemiripan (Sequence):** {sim:.2%}")

    st.markdown("**📊 Akurasi per Digit:**")
    for pos, val in perpos.items():
        st.markdown(f"- {pos.upper()}: {val:.2%}")

    fig = plot_rolling_accuracy(df)
    if fig:
        st.pyplot(fig)
