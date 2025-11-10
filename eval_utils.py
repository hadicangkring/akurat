# eval_utils.py
# Modul pendukung untuk evaluasi kinerja model Markov Fusion Deluxe

import pandas as pd
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
import os

# === 1. Fungsi: Baca log hasil prediksi ===
def load_log(path="data/prediksi_log.csv"):
    """Membaca file log prediksi jika sudah pernah disimpan."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if {"prediksi_4digit", "real_4digit"}.issubset(df.columns):
            return df
    except Exception:
        pass
    return pd.DataFrame()

# === 2. Fungsi: Hitung tingkat kesamaan penuh ===
def exact_match_rate(df):
    """Menghitung persentase prediksi yang 100% sama dengan hasil real."""
    if df.empty:
        return 0
    return (df["prediksi_4digit"].astype(str) == df["real_4digit"].astype(str)).mean()

# === 3. Fungsi: Hitung akurasi per posisi digit ===
def per_position_accuracy(df):
    """Menghitung akurasi per digit posisi (1–4)."""
    if df.empty:
        return {}
    acc = {}
    for pos in range(4):
        correct = df.apply(
            lambda r: str(r["prediksi_4digit"]).zfill(4)[pos]
            == str(r["real_4digit"]).zfill(4)[pos],
            axis=1,
        )
        acc[f"pos{pos+1}"] = correct.mean()
    return acc

# === 4. Fungsi: Hitung kemiripan urutan (Sequence Similarity) ===
def avg_similarity(df):
    """Rata-rata kemiripan urutan (0–1) antara prediksi dan real."""
    if df.empty:
        return 0
    return df.apply(
        lambda r: SequenceMatcher(
            None, str(r["prediksi_4digit"]).zfill(4), str(r["real_4digit"]).zfill(4)
        ).ratio(),
        axis=1,
    ).mean()

# === 5. Fungsi: Buat grafik akurasi bergulir (rolling accuracy) ===
def plot_rolling_accuracy(df, window=20):
    """Membuat grafik tren akurasi prediksi historis."""
    if df.empty:
        return None

    df = df.copy()
    df["correct"] = (
        df["prediksi_4digit"].astype(str) == df["real_4digit"].astype(str)
    ).astype(int)
    df["rolling_acc"] = df["correct"].rolling(window=window, min_periods=5).mean()

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df.index, df["rolling_acc"], color="tab:blue", linewidth=2)
    ax.set_title("📈 Tren Akurasi Prediksi (Top-1)")
    ax.set_xlabel("Prediksi ke-")
    ax.set_ylabel("Akurasi Bergulir")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    return fig

# === 6. Fungsi utama untuk menampilkan semua metrik di Streamlit ===
def tampilkan_evaluasi(st, log_path="data/prediksi_log.csv"):
    """
    Menampilkan hasil evaluasi historis langsung di halaman Streamlit.
    """
    st.subheader("📈 Evaluasi Kinerja Model Historis")

    df = load_log(log_path)
    if df.empty:
        st.info("Belum ada data prediksi yang tersimpan untuk evaluasi.")
        return

    # Hitung semua metrik
    exact = exact_match_rate(df)
    perpos = per_position_accuracy(df)
    sim = avg_similarity(df)

    # Tampilkan hasil numerik
    st.markdown(f"**🎯 Top-1 Exact Match:** {exact:.2%}")
    st.markdown(f"**🧩 Rata-rata Kemiripan Urutan:** {sim:.2%}")

    st.markdown("**📊 Akurasi per Digit:**")
    for pos, val in perpos.items():
        st.markdown(f"- {pos.upper()}: {val:.2%}")

    # Tampilkan grafik tren
    fig = plot_rolling_accuracy(df)
    if fig:
        st.pyplot(fig)
